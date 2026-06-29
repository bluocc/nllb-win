from optimum.intel.openvino import OVModelForSeq2SeqLM
from transformers import AutoTokenizer

from src.config import MODEL_NAME, OPENVINO_MODEL_NAME

OUTPUT_DIR = OPENVINO_MODEL_NAME

print("加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("导出 OpenVINO 模型...")
model = OVModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    export=True,
)

print("保存模型...")
model.save_pretrained(OUTPUT_DIR)

print("保存 Tokenizer...")
tokenizer.save_pretrained(OUTPUT_DIR)

print("导出完成！")