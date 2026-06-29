from optimum.intel.openvino import OVModelForSeq2SeqLM
from transformers import AutoTokenizer

from src.config import MODEL_NAME, OPENVINO_MODEL_NAME


def export_model():
    print("加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("导出 OpenVINO 模型...")
    model = OVModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        export=True,
    )

    print("保存模型...")
    model.save_pretrained(OPENVINO_MODEL_NAME)

    print("保存 Tokenizer...")
    tokenizer.save_pretrained(OPENVINO_MODEL_NAME)

    print("导出完成！")


if __name__ == "__main__":
    export_model()
