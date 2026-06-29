from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

from src.config import CACHE_DIR
from src.config import MODEL_NAME


def download_model():
    """下载模型和 tokenizer"""

    print(f"开始下载模型：{MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
    )

    print("模型下载完成！")

    return tokenizer, model


if __name__ == "__main__":
    download_model()