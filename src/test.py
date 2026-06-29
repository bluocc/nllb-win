import openvino as ov
core = ov.Core()
for d in core.available_devices:
    print(d, '->', core.get_property(d, 'FULL_DEVICE_NAME'))