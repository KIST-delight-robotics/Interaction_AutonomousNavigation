import tensorrt as trt

FIXED_PRECISION_LAYERS = [
    "/model/model/transformer_module/decoder/layers.1/cross_attn/MatMul_4",
    "/model/model/transformer_module/decoder/layers.2/cross_attn/MatMul_4",
    "/model/model/transformer_module/decoder/layers.3/cross_attn/MatMul_4",
    "/model/model/transformer_module/decoder/layers.4/cross_attn/MatMul_4",
    "/model/model/transformer_module/decoder/layers.5/cross_attn/MatMul_4",
    "/model/model/transformer_module/decoder/layers.6/cross_attn/MatMul_4",
    "/model/model/transformer_module/decoder/layers.7/cross_attn/MatMul_4",
    "/model/model/transformer_module/decoder/layers.8/cross_attn/MatMul_4"
]


def set_fixed_precision(network, fixed_precision_layers=FIXED_PRECISION_LAYERS, precision=trt.DataType.FLOAT):
    """
    network: TensorRT network
    fixed_precision_layers: 강제로 precision을 고정할 layer 이름 리스트
    precision: 강제로 지정할 trt.DataType (예: trt.DataType.FLOAT, trt.DataType.HALF)
    """
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for fixed_precision_layer in fixed_precision_layers:
            if layer.name == fixed_precision_layer:
                layer.precision = precision  # 연산 자체 FP32로 강제
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, precision)  # 출력도 FP32로 강제
