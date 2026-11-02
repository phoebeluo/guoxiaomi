"""把 `data/sessions/*.json` 迁进关系库（2026-11-02 · M14-C）。

    python tools/migrate_json_to_db.py --dry-run     # 只报数，不写库
    python tools/migrate_json_to_db.py               # 迁移，已存在的跳过
    python tools/migrate_json_to_db.py --overwrite   # 覆盖库里同名会话

## 三个刻意的设计

1. **幂等**：默认跳过库里已存在的会话。迁移脚本会被重复执行
   （改一次代码跑一次），不幂等就会把同一份会话写成两份，
   轮次唯一约束会直接报错，或者更糟——静默覆盖掉新数据。
2. **dry-run 先看数**：迁移前先知道「要迁几个、每个多少条、有没有解析失败的」，
   而不是先写再查。解析失败的文件名要单独列出来，因为那说明 JSON 结构本身有问题。
3. **不删源文件**：迁移只做「复制到库里」，JSON 原样保留。
   删源文件是独立的一次动作，得在库核对完之后再做——
   否则迁移一错，原始数据就没了。
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core  # noqa: E402
from db import init_db, store  # noqa: E402


def scan(sessions_dir):
    """读一遍所有 json，返回 (可迁移列表, 解析失败列表)。"""
    ok, bad = [], []
    for path in sorted(glob.glob(os.path.join(sessions_dir, "*.json"))):
        name = os.path.basename(path)[:-5]
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            msgs = data.get("messages") or []
            ok.append({
                "name": name,
                "nickname": data.get("nickname", "果小蜜"),
                "character": data.get("character", "热情、专业、懂生活、像闺蜜一样贴心"),
                "messages": msgs,
            })
        except Exception as e:  # noqa: BLE001
            bad.append((name, f"{type(e).__name__}: {e}"))
    return ok, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions-dir", default=core.SESSIONS_DIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ok, bad = scan(args.sessions_dir)
    total_msgs = sum(len(x["messages"]) for x in ok)
    print(f"扫描目录 : {args.sessions_dir}")
    print(f"可迁移   : {len(ok)} 个会话 / {total_msgs} 条消息")
    if bad:
        print(f"解析失败 : {len(bad)} 个")
        for name, err in bad:
            print(f"  - {name}: {err}")

    if args.dry_run:
        print("\n[dry-run] 未写库。去掉 --dry-run 执行迁移。")
        return 0

    init_db()
    existing = set(store.list_session_names())
    moved, skipped, failed = 0, 0, 0
    for item in ok:
        if item["name"] in existing and not args.overwrite:
            skipped += 1
            continue
        try:
            store.upsert_session(item["name"], item["messages"],
                                 item["nickname"], item["character"])
            moved += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  迁移失败 {item['name']}: {type(e).__name__}: {e}")

    print(f"\n迁移完成 : 写入 {moved} / 跳过 {skipped} / 失败 {failed}")
    print(f"库内统计 : {store.stats()}")
    print(f"库文件   : {os.environ.get('GUOXIAOMI_DB') or 'data/guoxiaomi.db'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
