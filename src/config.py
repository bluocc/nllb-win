"""
项目配置
"""

# Hugging Face 模型
MODEL_NAME = "facebook/nllb-200-distilled-1.3B"

# OpenVINO 模型路径
OPENVINO_MODEL_NAME = "models/openvino"

# OpenVINO 推理设备
DEVICE = "GPU"

# 固定翻译成中文
TARGET_LANGUAGE = "zho_Hans"

# 模型缓存目录（None 表示使用 Hugging Face 默认缓存）
CACHE_DIR = None

LANGUAGE_MAP = {
    "en": "eng_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "zh-cn": "zho_Hans",
}
#OpenVINO是否重新导出
EXPORT_OPENVINO = False
