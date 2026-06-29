from langdetect import detect

from src.config import LANGUAGE_MAP


def detect_language(text: str) -> str:
    lang = detect(text)
    return LANGUAGE_MAP.get(lang)


if __name__ == "__main__":
    print(detect_language("吃饭了吗"))