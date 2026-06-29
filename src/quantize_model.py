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


def _copy_ignore_errors(src_dir, dst_dir):
    try:
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True, ignore_dangling_symlinks=True)
    except Exception as e:
        print(f"  警告：复制 {src_dir} 时出错（{e}），跳过...")
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            try:
                shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            except Exception:
                pass


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

    print("  量化 encoder（INT8）...")
    encoder_xml = os.path.join(OPENVINO_MODEL_NAME, "openvino_encoder_model.xml")
    ov_encoder = core.read_model(encoder_xml)
    calib_data = _make_calib_data(ov_encoder, tokenizer, CALIBRATION_SENTENCES)
    quantized_encoder = nncf.quantize(
        ov_encoder,
        nncf.Dataset(calib_data),
        model_type=nncf.ModelType.TRANSFORMER,
        subset_size=10,
    )
    out_xml = os.path.join(OPENVINO_INT8_NAME, "openvino_encoder_model.xml")
    out_bin = os.path.join(OPENVINO_INT8_NAME, "openvino_encoder_model.bin")
    ov.serialize(quantized_encoder, out_xml, out_bin)
    print("    ✓ encoder 量化完成")

    print("  复制 decoder（保持 FP16）...")
    for ext in [".xml", ".bin"]:
        shutil.copy2(
            os.path.join(OPENVINO_MODEL_NAME, f"openvino_decoder_model{ext}"),
            os.path.join(OPENVINO_INT8_NAME, f"openvino_decoder_model{ext}"),
        )
    print("    ✓ decoder 复制完成")

    print("  复制配置文件...")
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
        _copy_ignore_errors(encoder_sub, os.path.join(OPENVINO_INT8_NAME, "encoder"))

    print("  INT8 量化全部完成！")


if __name__ == "__main__":
    quantize_model()
