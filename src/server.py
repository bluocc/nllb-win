import os
import time
import shutil

from flask import Flask, request, jsonify, render_template

from src.translator import Translator
from src.detector import detect_language
from src.config import OPENVINO_MODEL_NAME, OPENVINO_INT8_NAME

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
fp16_translator = None
int8_translator = None


def _get_source_lang(text, source_lang):
    if not text:
        return None, "请输入要翻译的文本"
    if source_lang == "auto":
        detected = detect_language(text)
        if detected is None:
            return None, "无法自动检测语言，请手动指定源语言"
        return detected, None
    return source_lang, None


def get_fp16(model_path=OPENVINO_MODEL_NAME):
    global fp16_translator
    if fp16_translator is None and os.path.isdir(model_path):
        fp16_translator = Translator(model_path)
    return fp16_translator


def get_int8():
    global int8_translator
    if int8_translator is None and os.path.isdir(OPENVINO_INT8_NAME):
        int8_translator = Translator(OPENVINO_INT8_NAME)
    return int8_translator


@app.route("/")
def index():
    return render_template("index.html", languages=LANGUAGE_NAMES)


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text", "").strip()
    source_lang = data.get("source_lang", "auto")

    source_lang, err = _get_source_lang(text, source_lang)
    if err:
        return jsonify({"error": err}), 400

    t = get_fp16()
    if t is None:
        return jsonify({"error": "FP16 模型未加载"}), 500

    result = t.translate(text, source_lang)
    return jsonify({"translation": result})


@app.route("/languages")
def languages():
    return jsonify(LANGUAGE_NAMES)


@app.route("/compare", methods=["POST"])
def compare():
    data = request.get_json()
    text = data.get("text", "").strip()
    source_lang = data.get("source_lang", "auto")

    source_lang, err = _get_source_lang(text, source_lang)
    if err:
        return jsonify({"error": err}), 400

    results = {}
    for model_type, fn in [("fp16", get_fp16), ("int8", get_int8)]:
        t = fn()
        if t is None:
            results[model_type] = None
        else:
            start = time.time()
            r = t.translate(text, source_lang)
            elapsed = time.time() - start
            results[model_type] = {
                "translation": r,
                "time": round(elapsed, 3),
            }

    return jsonify(results)


@app.route("/model-status")
def model_status():
    fp16_ok = os.path.isdir(OPENVINO_MODEL_NAME) and os.path.isfile(
        os.path.join(OPENVINO_MODEL_NAME, "openvino_encoder_model.xml")
    )
    int8_ok = os.path.isdir(OPENVINO_INT8_NAME) and os.path.isfile(
        os.path.join(OPENVINO_INT8_NAME, "openvino_encoder_model.xml")
    )
    return jsonify({
        "fp16": fp16_ok,
        "int8": int8_ok,
    })


@app.route("/delete-model", methods=["POST"])
def delete_model():
    data = request.get_json()
    model_type = data.get("model")

    if model_type == "fp16":
        path = OPENVINO_MODEL_NAME
    elif model_type == "int8":
        path = OPENVINO_INT8_NAME
    else:
        return jsonify({"error": "无效的模型类型，请指定 fp16 或 int8"}), 400

    if not os.path.isdir(path):
        return jsonify({"error": "模型目录不存在"}), 400

    shutil.rmtree(path)
    global fp16_translator, int8_translator
    if model_type == "fp16":
        fp16_translator = None
    else:
        int8_translator = None

    return jsonify({"success": True, "deleted": model_type})
