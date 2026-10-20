"""BCI Competition IV-2a 数据下载脚本

通过 MOABB 库下载数据，转存为与竞赛格式兼容的 .mat 文件。

用法：
    直接修改下方参数区后运行：python download_data.py
"""

import sys
from pathlib import Path

# ============================================================
# 参数配置（修改此处）
# ============================================================

SAVE_DIR = "data/raw"   # 数据保存目录

# ============================================================
# 以下为正式代码，不含可调参数
# ============================================================


def main():
    # 检查依赖
    missing = []
    try:
        import moabb
    except ImportError:
        missing.append("moabb")
    try:
        import mne
    except ImportError:
        missing.append("mne")
    try:
        import scipy
    except ImportError:
        missing.append("scipy")

    if missing:
        print(f"❌ 缺少依赖，请先安装：pip install {' '.join(missing)}")
        sys.exit(1)

    save_path = Path(SAVE_DIR)
    save_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BCI Competition IV-2a 数据下载")
    print(f"保存目录: {save_path.resolve()}")
    print("=" * 60)

    # 检查已有文件
    existing = list(save_path.glob("A0*.mat"))
    if len(existing) == 18:
        print("✅ 所有 18 个数据文件已存在（9名被试 × 2个Session），跳过下载。")
        return
    elif existing:
        print(f"发现 {len(existing)} 个已有文件，将跳过已存在的文件。")

    # 执行下载
    from mi_eeg.data.bci import download_bci2a
    download_bci2a(save_dir=SAVE_DIR)

    # 验证结果
    mat_files = list(save_path.glob("A0*.mat"))
    print(f"\n下载完成，共 {len(mat_files)} 个文件：")
    for f in sorted(mat_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}  ({size_mb:.1f} MB)")

    if len(mat_files) < 18:
        print(f"\n⚠️  预期 18 个文件，实际 {len(mat_files)} 个，请检查下载日志。")
    else:
        print("\n✅ 所有文件下载完成，可运行实验。")


if __name__ == "__main__":
    main()
