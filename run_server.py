#!/usr/bin/env python3
import os
import sys
import subprocess
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.absolute()
VENV_DIR = ROOT_DIR / ".venv"


def in_venv():
    return sys.prefix != sys.base_prefix


def setup_venv_and_restart():
    if VENV_DIR.exists():
        python = VENV_DIR / ("Scripts" if os.name == "nt" else "bin") / "python"
    else:
        print("=" * 50)
        print("  NLLB 翻译服务 —— 首次启动")
        print("=" * 50)
        print("\n[1/3] 创建虚拟环境...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        python = VENV_DIR / ("Scripts" if os.name == "nt" else "bin") / "python"

        print("[2/3] 安装依赖（首次安装可能需要几分钟）...")
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(ROOT_DIR / "requirements.txt")],
            check=True,
        )

        print("[3/3] 依赖安装完成！\n")

    os.chdir(ROOT_DIR)
    os.execl(str(python), str(python), *sys.argv)


def check_model():
    from src.config import MODEL_NAME, CACHE_DIR
    from transformers import AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        print("  ✓ 模型已下载")
    except Exception:
        print("  ↓ 模型未下载，开始下载（首次需要较长时间）...")
        from src.download_model import download_model

        download_model()


def check_export():
    from src.config import OPENVINO_MODEL_NAME

    xml_path = os.path.join(OPENVINO_MODEL_NAME, "openvino_encoder_model.xml")
    if os.path.isfile(xml_path):
        print("  ✓ OpenVINO 模型已编译")
    else:
        print("  ↓ OpenVINO 模型未编译，开始导出...")
        from src.export_model import export_model

        export_model()


def check_quantize():
    from src.config import OPENVINO_INT8_NAME

    xml_path = os.path.join(OPENVINO_INT8_NAME, "openvino_encoder_model.xml")
    if os.path.isfile(xml_path):
        print("  ✓ INT8 量化模型已就绪")
    else:
        print("  ↓ INT8 模型未量化，开始量化...")
        from src.quantize_model import quantize_model

        quantize_model()


def main():
    if not in_venv():
        setup_venv_and_restart()

    os.chdir(ROOT_DIR)

    print("=" * 50)
    print("  NLLB 翻译服务")
    print("=" * 50)
    print()

    print("[1/3] 检查模型...")
    check_model()

    print("[2/3] 检查 OpenVINO 模型...")
    check_export()

    print("[3/3] 检查 INT8 量化模型...")
    check_quantize()

    print()
    print("-" * 50)
    print("  启动服务器：http://localhost:5000")
    print("-" * 50)
    print()

    from src.server import app

    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
