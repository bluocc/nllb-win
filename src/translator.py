from optimum.intel.openvino import OVModelForSeq2SeqLM
from transformers import AutoTokenizer

from src.config import MODEL_NAME, OPENVINO_MODEL_NAME, TARGET_LANGUAGE
from src.config import DEVICE
from src.detector import detect_language


class Translator:

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = OPENVINO_MODEL_NAME
        print(f"正在加载 Tokenizer ({model_path})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        print(f"正在加载 OpenVINO 模型 ({model_path})...")
        self.model = OVModelForSeq2SeqLM.from_pretrained(
            model_path,
            device=DEVICE,
        )

        print("模型加载完成！")
        print(f"  模型加载设备: {self.model._device}")

    def translate(self, text: str, source_lang: str):

        self.tokenizer.src_lang = source_lang

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        )

        generated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(
                TARGET_LANGUAGE
            ),
        )

        result = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return result[0]


if __name__ == "__main__":
    translator = Translator()
    print(f"  模型加载设备: {translator.model._device}")
