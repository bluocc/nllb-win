import os
import shutil
import numpy as np

import openvino as ov
import nncf
from transformers import AutoTokenizer

from src.config import OPENVINO_MODEL_NAME, OPENVINO_INT8_NAME

CALIBRATION_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "Please translate this sentence from English to Chinese.",
    "Machine learning models require large amounts of data.",
    "The weather today is sunny and warm.",
    "I would like to book a flight to Paris.",
    "Natural language processing is a field of AI.",
    "The capital of France is Paris.",
    "Deep learning has achieved great success recently.",
    "Could you please help me with this problem?",
]


def _make_calib_data(model, tokenizer, sentences, max_seq_len=64):
    inputs = model.inputs

    calib_data = []
    for s in sentences[:10]:
        tokens = tokenizer(
            s,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
        )
        feed_dict = {}
        for inp in inputs:
            name = inp.any_name
            shape = []
            for dim in inp.partial_shape:
                shape.append(dim.get_length() if dim.is_static else max_seq_len)
            shape = tuple(max(1, int(s)) for s in shape)

            if inp.element_type == ov.Type.i32:
                if "input_ids" in name:
                    feed_dict[name] = tokens["input_ids"].astype(np.int32)
                elif "attention_mask" in name or "mask" in name.lower():
                    feed_dict[name] = tokens["attention_mask"].astype(np.int32)
                else:
                    feed_dict[name] = np.zeros(shape, dtype=np.int32)
            else:
                feed_dict[name] = np.zeros(shape, dtype=np.float32)

        calib_data.append(feed_dict)

    return calib_data


def quantize_model():
    if not os.path.isdir(OPENVINO_MODEL_NAME):
        print("  FP16 模型不存在，请先导出 OpenVINO 模型")
        return

    os.makedirs(OPENVINO_INT8_NAME, exist_ok=True)

    print("  加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(OPENVINO_MODEL_NAME)

    core = ov.Core()
    components = ["encoder", "decoder"]

    for component in components:
        xml_path = os.path.join(
            OPENVINO_MODEL_NAME, f"openvino_{component}_model.xml"
        )
        if not os.path.isfile(xml_path):
            print(f"  跳过 {component}：{xml_path} 不存在")
            continue

        print(f"  读取 {component} 模型...")
        ov_model = core.read_model(xml_path)

        print(f"  生成 {component} 校准数据（10 条样本）...")
        calib_data = _make_calib_data(ov_model, tokenizer, CALIBRATION_SENTENCES)

        print(f"  量化 {component}（INT8）...")
        quantized_model = nncf.quantize(
            ov_model,
            nncf.Dataset(calib_data),
            model_type=nncf.ModelType.TRANSFORMER,
            subset_size=10,
        )

        out_xml = os.path.join(
            OPENVINO_INT8_NAME, f"openvino_{component}_model.xml"
        )
        out_bin = os.path.join(
            OPENVINO_INT8_NAME, f"openvino_{component}_model.bin"
        )
        print(f"  保存 {component}...")
        ov.serialize(quantized_model, out_xml, out_bin)
        print(f"    ✓ {component} 量化完成")

    for f in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        src = os.path.join(OPENVINO_MODEL_NAME, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(OPENVINO_INT8_NAME, f))

    encoder_sub = os.path.join(OPENVINO_MODEL_NAME, "encoder")
    if os.path.isdir(encoder_sub):
        dst = os.path.join(OPENVINO_INT8_NAME, "encoder")
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(encoder_sub):
            shutil.copy2(os.path.join(encoder_sub, f), os.path.join(dst, f))

    print("  INT8 量化全部完成！")


if __name__ == "__main__":
    quantize_model()
