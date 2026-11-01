"""pytest 公共配置：把项目根目录放进 sys.path，让测试能 import rag.*。"""

import os
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)
