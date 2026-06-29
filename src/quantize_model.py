import os
import shutil
import numpy as np

import openvino as ov
import nncf

from src.config import OPENVINO_MODEL_NAME, OPENVINO_INT8_NAME


def _generate_calibration_data(model_xml, num_samples=50, max_seq_len=128):
    core = ov.Core()
    model = core.read_model(model_xml)
    inputs = model.inputs
    calib_data = []
    for _ in range(num_samples):
        feed_dict = {}
        for inp in inputs:
            partial_shape = inp.partial_shape
            shape = []
            for dim in partial_shape:
                if dim.is_static:
                    shape.append(dim.get_length())
                else:
                    shape.append(max_seq_len)
            shape = [max(1, s) for s in shape]
            shape = tuple(int(s) for s in shape)
            if inp.element_type == ov.Type.i32:
                feed_dict[inp.any_name] = np.random.randint(
                    0, 100, shape
                ).astype(np.int32)
            else:
                feed_dict[inp.any_name] = np.random.randn(*shape).astype(
                    np.float32
                )
        calib_data.append(feed_dict)
    return calib_data


def quantize_model():
    if not os.path.isdir(OPENVINO_MODEL_NAME):
        print("  FP16 模型不存在，请先导出 OpenVINO 模型")
        return

    os.makedirs(OPENVINO_INT8_NAME, exist_ok=True)

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

        print(f"  生成 {component} 校准数据...")
        calib_data = _generate_calibration_data(xml_path)

        print(f"  量化 {component}（INT8）...")
        quantized_model = nncf.quantize(
            ov_model,
            nncf.Dataset(calib_data),
            model_type=nncf.ModelType.TRANSFORMER,
            subset_size=min(len(calib_data), 50),
        )

        out_xml = os.path.join(
            OPENVINO_INT8_NAME, f"openvino_{component}_model.xml"
        )
        print(f"  保存 {component}...")
        core.save_model(quantized_model, out_xml)
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
            shutil.copy2(
                os.path.join(encoder_sub, f), os.path.join(dst, f)
            )

    print("  INT8 量化全部完成！")


if __name__ == "__main__":
    quantize_model()
