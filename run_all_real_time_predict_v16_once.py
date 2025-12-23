import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    # 当前目录
    root = Path(__file__).resolve().parent

    # 所有需要执行的 v16 预测脚本（按文件名排序，便于查看）
    scripts = sorted(
        [
            "real_time_predict_v16_000625.py",
            "real_time_predict_v16_002025.py",
            "real_time_predict_v16_002241.py",
            "real_time_predict_v16_002266.py",
            "real_time_predict_v16_002475.py",
            "real_time_predict_v16_002706.py",
            "real_time_predict_v16_002837.py",
            "real_time_predict_v16_002851.py",
            "real_time_predict_v16_300153.py",
            "real_time_predict_v16_300274.py",
            "real_time_predict_v16_300499.py",
            "real_time_predict_v16_300726.py",
            "real_time_predict_v16_300762.py",
            "real_time_predict_v16_301005.py",
            "real_time_predict_v16_600730.py",
            "real_time_predict_v16_600733.py",
            "real_time_predict_v16_601399.py",
            "real_time_predict_v16_603267.py",
            "real_time_predict_v16_603698.py",
            # 新增：道通科技688208（A股）和道通转债118013（可转债准实时）
            "real_time_predict_v16_688208.py",
            "real_time_predict_v16_118013.py",
        ]
    )

    # 日志文件（按日期命名）
    now = datetime.now()
    log_name = f"real_time_predict_v16_all_{now.strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = root / log_name

    python_exe = sys.executable  # 使用当前环境的 Python

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(
            f"统一执行 real_time_predict_v16_* 预测脚本，仅执行各脚本的首次预测输出\n"
            f"时间：{now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Python：{python_exe}\n"
            f"脚本总数：{len(scripts)}\n"
            f"{'-' * 80}\n\n"
        )

        for idx, script in enumerate(scripts, start=1):
            script_path = root / script

            log_file.write(f"[{idx}/{len(scripts)}] 开始执行：{script}\n")
            log_file.write(f"路径：{script_path}\n")
            log_file.write("-" * 80 + "\n")
            log_file.flush()

            if not script_path.exists():
                msg = f"文件不存在：{script_path}\n"
                print(msg, end="")
                log_file.write(msg + "\n")
                log_file.write("=" * 80 + "\n\n")
                log_file.flush()
                continue

            try:
                # 直接以子进程方式运行脚本，捕获标准输出和错误输出
                # 注意：这里不会对脚本内部逻辑做任何修改，
                # 假设每个脚本运行一次就会产生一次预测并退出。
                result = subprocess.run(
                    [python_exe, str(script_path)],
                    cwd=str(root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

                # 写入标准输出
                if result.stdout:
                    log_file.write("【标准输出】\n")
                    log_file.write(result.stdout)
                    if not result.stdout.endswith("\n"):
                        log_file.write("\n")

                # 写入错误输出
                if result.stderr:
                    log_file.write("\n【错误输出】\n")
                    log_file.write(result.stderr)
                    if not result.stderr.endswith("\n"):
                        log_file.write("\n")

                log_file.write("=" * 80 + "\n\n")
                log_file.flush()

                # 同时也简单打印到当前控制台，方便你实时观察
                print(f"[{idx}/{len(scripts)}] {script} 运行结束，返回码：{result.returncode}")

            except Exception as e:
                err_msg = f"运行 {script} 发生异常：{e}\n"
                print(err_msg, end="")
                log_file.write("\n【运行异常】\n")
                log_file.write(err_msg)
                log_file.write("=" * 80 + "\n\n")
                log_file.flush()

    print(f"\n全部脚本执行完毕，输出已写入：{log_path}")


if __name__ == "__main__":
    main()


