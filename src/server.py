from flask import Flask, request, jsonify, render_template

from src.translator import Translator
from src.detector import detect_language

LANGUAGE_NAMES = {
    "auto": "自动检测",
    "en": "英文",
    "ja": "日文",
    "ko": "韩文",
    "fr": "法文",
    "de": "德文",
    "zh-cn": "中文",
}

app = Flask(__name__)
translator = Translator()


@app.route("/")
def index():
    return render_template("index.html", languages=LANGUAGE_NAMES)


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text", "").strip()
    source_lang = data.get("source_lang", "auto")

    if not text:
        return jsonify({"error": "请输入要翻译的文本"}), 400

    if source_lang == "auto":
        detected = detect_language(text)
        if detected is None:
            return jsonify({"error": "无法自动检测语言，请手动指定源语言"}), 400
        source_lang = detected

    result = translator.translate(text, source_lang)
    return jsonify({"translation": result})


@app.route("/languages")
def languages():
    return jsonify(LANGUAGE_NAMES)
