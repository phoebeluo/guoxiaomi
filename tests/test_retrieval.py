"""检索层单元测试：BM25 与语义近似。

为什么这些断言值得写
--------------------
`run_all.py` 里的 12 个脚本测的是「全量跑一遍有没有崩」，属于回归；
这里测的是**单个函数的契约**：分词切成什么、零命中返回什么、`b` 改了会不会影响 idf。
前者发现不了「改了 b 顺手把 idf 也算错了」这类问题，后者能。

断言全部用**手算得出的常量**，不是用另一个实现去对拍 —— 对拍只能证明两个实现一样错。
"""

import glob
import os

import pytest

from rag.bm25 import BM25, tokenize
from rag.vector import VectorIndex

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(PROJECT, "data", "docs")


@pytest.fixture(scope="module")
def docs():
    out = []
    for p in sorted(glob.glob(os.path.join(DOCS_DIR, "*.md"))):
        with open(p, encoding="utf-8") as f:
            out.append((os.path.basename(p), f.read()))
    return out


class TestTokenize:
    """分词契约：切成什么、有什么副作用。"""

    def test_cjk_bigram(self):
        # 「满多少免运费」5 个汉字 → 4 个 bigram，顺序保留
        assert tokenize("满多少免运费") == ["满多", "多少", "少免", "免运", "运费"]

    def test_single_cjk_kept_as_unigram(self):
        # 孤立单字不能被丢掉，否则「梨」这种查询会变成空
        assert tokenize("梨") == ["梨"]

    def test_numbers_not_split(self):
        # 极简实现会把「99」拆成两个「9」，这里必须保持完整
        toks = tokenize("满 99 元免运费")
        assert "99" in toks
        assert "9" not in toks

    def test_fullwidth_normalized(self):
        # 全角「９９」要能命中半角「99」——NFKC 归一化的作用
        assert "99" in tokenize("满９９元")

    def test_stop_bigram_dropped(self):
        # 「以及」在停用表里，不该进入 token
        assert "以及" not in tokenize("苹果以及香蕉")

    def test_empty(self):
        assert tokenize("") == []


class TestBM25:
    """BM25 契约：打分结构、零命中、参数隔离。"""

    def test_idf_hand_calculated(self, docs):
        """IDF = ln(1 + (N-df+0.5)/(df+0.5))，用 45 篇的真实 df 手算核对。"""
        import math

        idx = BM25(docs)
        n = idx.N
        assert n == 45
        df = len(idx.postings["运费"])
        expected = math.log(1 + (n - df + 0.5) / (df + 0.5))
        assert idx.idf["运费"] == pytest.approx(expected, rel=1e-9)

    def test_rare_token_weighs_more(self, docs):
        """稀有词权重必须高于常见词，否则 IDF 写反了。"""
        idx = BM25(docs)
        # 「多少」2 篇 vs 「运费」11 篇
        assert idx.idf["多少"] > idx.idf["运费"]

    def test_no_negative_idf(self, docs):
        """全库都出现的词权重 ≥ 0（+1 的作用），绝不能为负。"""
        idx = BM25(docs)
        assert min(idx.idf.values()) >= 0.0

    def test_b_does_not_touch_idf(self, docs):
        """b 只进打分，不该影响 idf —— 这是「复用同一索引扫 b」的前提。"""
        lo, hi = BM25(docs, b=0.5), BM25(docs, b=1.0)
        assert lo.idf == hi.idf

    def test_b_enters_the_formula(self, docs):
        """b 必须真的进打分公式：同一批文档得分要变。

        ⚠️ 不能断言「排序会变」—— 实测 b 从 0 调到 1，48 题里 46 题排序不变，
        这条「不敏感」是 `b_sweep` 的结论，把它写进测试会变成假失败。
        这里只验证 b 确实参与了计算。
        """
        lo, hi = BM25(docs, b=0.0), BM25(docs, b=1.0)
        q = "订单满多少免运费"
        s_lo, s_hi = dict(lo.search(q, 5)), dict(hi.search(q, 5))
        common = set(s_lo) & set(s_hi)
        assert common, "b=0 与 b=1 的 top5 完全没有交集，可疑"
        assert any(abs(s_lo[d] - s_hi[d]) > 1e-6 for d in common), "b 没进打分公式"

    def test_zero_hit_returns_empty(self, docs):
        """全是不在词表的词 → 空列表，不能抛异常、不能返回全库。"""
        idx = BM25(docs)
        assert idx.search("zzzqqqxxx999", top_k=3) == []

    def test_topk_respected(self, docs):
        idx = BM25(docs)
        assert len(idx.search("免运费", top_k=3)) <= 3

    def test_scores_descending(self, docs):
        idx = BM25(docs)
        scores = [s for _, s in idx.search("坏果怎么赔", top_k=5)]
        assert scores == sorted(scores, reverse=True)


class TestVectorIndex:
    """语义近似契约：维度截断、余弦有界、零命中。"""

    def test_dim_capped_by_doc_count(self, docs):
        """维度 = min(128, 文档数)。45 篇只能到 45 维 —— 这是 45 篇档表现最好的原因。"""
        idx = VectorIndex(docs, dim=128)
        assert idx.dim == min(128, len(docs)) == 45

    def test_cosine_bounded(self, docs):
        """余弦必须在 [-1, 1]。越界说明归一化漏了。"""
        idx = VectorIndex(docs, dim=32)
        for _, s in idx.search("免运费怎么算", top_k=5):
            assert -1.0 - 1e-6 <= s <= 1.0 + 1e-6

    def test_zero_hit_returns_empty(self, docs):
        idx = VectorIndex(docs, dim=32)
        assert idx.search("zzzqqqxxx999", top_k=3) == []

    def test_scores_descending(self, docs):
        idx = VectorIndex(docs, dim=32)
        scores = [s for _, s in idx.search("坏果怎么赔", top_k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_same_index_same_result(self, docs):
        """同一个索引查两次，结果必须一致（不能有随机性）。"""
        idx = VectorIndex(docs, dim=32)
        assert idx.search("车厘子保存", 3) == idx.search("车厘子保存", 3)


class TestRetrievalContract:
    """两条检索路径必须接口一致 —— app.py 靠这一点在两者间直接互换。"""

    def test_same_methods(self, docs):
        bm, vi = BM25(docs), VectorIndex(docs, dim=32)
        for idx in (bm, vi):
            assert callable(idx.search) and callable(idx.text)
            hits = idx.search("免运费", top_k=2)
            assert isinstance(hits, list) and hits
            name, _ = hits[0]
            assert isinstance(idx.text(name), str) and idx.text(name)
