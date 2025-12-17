"""
sync_turnover_v16_all.py

用途：
- 将 real_time_predict_v16_603698.py 中与“实时/近似换手率 + get_current_market_price”相关的实现，
  同步到其余所有 real_time_predict_v16_*.py 文件，保证逻辑一致。

说明：
- 同步范围为：
  1) 从 'portfolio_state_mtime' 行开始，到 'def create_portfolio_web_app' 定义之前结束。
  2) 这段中包含：
     - portfolio_state_mtime 定义
     - LAST_TURNOVER_CACHE / LAST_TURNOVER_APPROX_FLAG
     - get_current_market_price
     - get_realtime_turnover
"""

import os

TEMPLATE_FILE = "real_time_predict_v16_603698.py"

TARGET_FILES = [
    "real_time_predict_v16_002025.py",
    "real_time_predict_v16_002241.py",
    "real_time_predict_v16_603267.py",
    "real_time_predict_v16_601399.py",
    "real_time_predict_v16_600730.py",
    "real_time_predict_v16_300762.py",
    "real_time_predict_v16_300726.py",
    "real_time_predict_v16_300499.py",
    "real_time_predict_v16_300274.py",
    "real_time_predict_v16_300153.py",
    "real_time_predict_v16_002851.py",
    "real_time_predict_v16_002837.py",
    "real_time_predict_v16_002706.py",
    "real_time_predict_v16_002475.py",
    "real_time_predict_v16_002266.py",
    "real_time_predict_v16_301005.py",  # 也一并用模板覆盖，保持完全一致
]


def extract_template_block(path: str) -> str:
    """从模板文件中提取要同步的代码块。"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    start_marker = "portfolio_state_mtime ="
    end_marker = "def create_portfolio_web_app"

    start_idx = content.find(start_marker)
    if start_idx == -1:
        raise RuntimeError(f"在模板文件中找不到 start_marker: {start_marker}")

    end_idx = content.find(end_marker, start_idx)
    if end_idx == -1:
        raise RuntimeError(f"在模板文件中找不到 end_marker: {end_marker}")

    # 保留到 end_marker 之前的换行
    block = content[start_idx:end_idx]
    return block.rstrip() + "\n\n"


def sync_file(target_path: str, template_block: str) -> bool:
    """用模板 block 替换目标文件中对应的代码段。"""
    if not os.path.exists(target_path):
        print(f"⚠️  文件不存在，跳过: {target_path}")
        return False

    with open(target_path, "r", encoding="utf-8") as f:
        content = f.read()

    start_marker = "portfolio_state_mtime ="
    end_marker = "def create_portfolio_web_app"

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker, start_idx)

    if start_idx == -1 or end_idx == -1:
        print(f"⚠️  {target_path} 中找不到同步范围标记，跳过")
        return False

    # 保留 end_marker 及其后内容
    new_content = content[:start_idx] + template_block + content[end_idx:]

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"✅ 已同步: {target_path}")
    return True


def main():
    if not os.path.exists(TEMPLATE_FILE):
        print(f"❌ 模板文件不存在: {TEMPLATE_FILE}")
        return

    template_block = extract_template_block(TEMPLATE_FILE)

    ok = 0
    fail = 0
    for path in TARGET_FILES:
        if sync_file(path, template_block):
            ok += 1
        else:
            fail += 1

    print("=" * 60)
    print(f"同步完成: 成功 {ok} 个, 失败 {fail} 个")


if __name__ == "__main__":
    main()


