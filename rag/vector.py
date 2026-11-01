"""语义近似检索：TF-IDF → SVD 低秩分解 → 余弦相似（纯 numpy，零 API，零新依赖）。

⚠️ 诚实标注（2026-10-28 定，务必遵守）
------------------------------------
**这不是 embedding。** embedding 是用神经网络把文本映射到稠密向量，能捕捉「包邮 ≈ 免运费」
这种同义关系。本文件是对**词袋矩阵做低秩分解**，它捕捉的是「哪些词经常一起出现」，
捕捉不到真正的语义。

所以简历/README 口径只能是：
  ✅「稀疏检索（BM25）与低秩语义近似（SVD）的召回对比」
  ❌ 不得出现 embedding / 向量检索 / ANN / 向量库 字样
除非另花约 1 元真调一次 `text-embedding-v3` 做对照（见主轨计划 W7 诚实标注）。

它在系统里干什么
----------------
BM25 只认字面：问「运费怎么算」，它会因为「运费」这个词而把《售后汇总》排前面。
这一路换一种「像不像」的算法：不数单个词，而是把每篇文档压成 128 个数字，
再看这些数字靠近不靠近。

为什么用 128 维：文档数只有 346，128 已经是文档数的三分之一，再高就开始拟合噪声。
为什么在篇级而不是块级：346 篇 × 10 万词表的矩阵是 138 MB，还能算；
块级 2,562 块会变成 1 GB，纯 numpy 扛不住。且 D 策略（BM25）也是篇级，两者可比。

为什么不用 numpy.linalg.svd 直接算
----------------------------------
X 是 (346, 100000)。直接 SVD 的计算量约 346×100000×346 = 1.2e10 次浮点运算，
且要一次性分配巨大的中间矩阵。
改用等价做法：对 X·Xᵀ（只有 346×346）做特征分解，右奇异向量由 Xᵀ·U/S 反推。
数学上完全等价，计算量一样但内存小两个数量级。

用法
----
    from rag.vector import VectorIndex
    idx = VectorIndex(docs)          # docs = [(doc_id, text)]
    idx.search("免运费怎么算", k=3)   # -> [(doc_id, 余弦), ...]
"""

import math
import time
from collections import Counter

import numpy as np

try:
    from rag.bm25 import tokenize
except ModuleNotFoundError:      # 支持 `python rag/vector.py` 直接跑（不走包路径）
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rag.bm25 import tokenize


class VectorIndex:
    """低秩语义近似索引。文档 = (doc_id, text) 列表。"""

    def __init__(self, docs, dim=128, verbose=False):
        t0 = time.perf_counter()
        self.doc_ids = [d for d, _ in docs]
        self._texts = [t for _, t in docs]
        n = len(docs)

        # ---- 1. 分词 + 建词表 ----
        tokenized = [tokenize(t) for _, t in docs]
        vocab = {}
        for toks in tokenized:
            for tk in set(toks):
                if tk not in vocab:
                    vocab[tk] = len(vocab)
        self.vocab = vocab
        V = len(vocab)

        # ---- 2. TF-IDF 加权 ----
        # df：每个 token 出现在多少篇文档里
        df = np.zeros(V, dtype=np.float32)
        for toks in tokenized:
            for tk in set(toks):
                df[vocab[tk]] += 1
        # 与 BM25 用同一个 IDF 式子，保证两路只在「相似度算法」上不同，
        # 不在「词的权重定义」上不同 —— 否则差异无法归因。
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)

        X = np.zeros((n, V), dtype=np.float32)
        for i, toks in enumerate(tokenized):
            for tk, c in Counter(toks).items():
                X[i, vocab[tk]] = c * idf[vocab[tk]]
            # 行内 L2 归一化：让长短文档可比，否则又是「长文霸榜」
            norm = float(np.linalg.norm(X[i]))
            if norm > 0:
                X[i] /= norm

        # ---- 3. 低秩分解 ----
        dim = min(dim, min(n, V))
        G = X @ X.T                                  # (n, n)，对称半正定
        vals, vecs = np.linalg.eigh(G)                # 特征值升序
        order = np.argsort(vals)[::-1][:dim]
        S = np.sqrt(np.maximum(vals[order], 0))       # 奇异值
        U = vecs[:, order]                            # (n, dim)
        safe = np.where(S > 1e-8, S, 1.0)
        # 右奇异向量 V = Xᵀ·U/S，转置后是 (dim, V)，用于把查询投影进同一空间
        self.Vt = (X.T @ U / safe).T.astype(np.float32)

        # 文档向量 = 左奇异向量 × 奇异值（等价于 X @ Vᵀ）
        dvec = (U * S).astype(np.float32)
        # 再归一化：余弦只看方向
        dn = np.linalg.norm(dvec, axis=1, keepdims=True)
        self.doc_vecs = dvec / np.where(dn > 0, dn, 1.0)

        self.idf = idf
        self.dim = dim
        self.N = n
        self.build_s = time.perf_counter() - t0
        if verbose:
            print(f"向量索引：{n} 篇 × 词表 {V} → {dim} 维，"
                  f"构建 {self.build_s:.2f}s，占比 {dim / max(n, 1):.0%}")

    def _query_vec(self, query):
        """把查询词按同一套 IDF 加权，再投影到 128 维。"""
        q = np.zeros(len(self.vocab), dtype=np.float32)
        for tk, c in Counter(tokenize(query)).items():
            j = self.vocab.get(tk)
            if j is not None:
                q[j] = c * self.idf[j]
        nq = float(np.linalg.norm(q))
        if nq == 0:
            return None
        q /= nq
        v = self.Vt @ q
        nv = float(np.linalg.norm(v))
        return v / nv if nv > 0 else None

    def search(self, query, top_k=5):
        """返回 [(doc_id, 余弦)]，按余弦降序。余弦范围 [-1, 1]，有界。"""
        v = self._query_vec(query)
        if v is None:                      # 查询里一个词都不在词表 → 无结果
            return []
        sims = self.doc_vecs @ v
        order = np.argsort(sims)[::-1][:top_k]
        return [(self.doc_ids[i], float(sims[i])) for i in order]

    def text(self, doc_id):
        return self._texts[self.doc_ids.index(doc_id)]


if __name__ == "__main__":
    import glob
    import os

    docs = []
    for d in ["data/docs", "data/raw/external_clean"]:
        for p in sorted(glob.glob(os.path.join(d, "*.md"))):
            with open(p, encoding="utf-8") as f:
                docs.append((os.path.basename(p), f.read()))

    idx = VectorIndex(docs, verbose=True)

    t0 = time.perf_counter()
    for _ in range(100):
        idx.search("满多少免运费")
    per = (time.perf_counter() - t0) / 100 * 1000
    print(f"单次查询 {per:.2f} ms（346 篇档）")

    for q in ["满多少免运费", "坏果怎么赔", "车厘子怎么保存"]:
        print(f"\nQ: {q}")
        for name, sim in idx.search(q, top_k=3):
            print(f"  {sim:6.3f}  {name}")
