# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import onnx


def _copy_value_info(value_info: onnx.ValueInfoProto) -> onnx.ValueInfoProto:
    """
    Create a copy of an ONNX ValueInfoProto with the same name, type, and shape.
    Args:
        value_info: The ValueInfoProto to copy.
    Returns:
        A copy of the input ValueInfoProto.
    """
    return onnx.helper.make_tensor_value_info(
        value_info.name,
        value_info.type.tensor_type.elem_type,
        [d.dim_value if d.dim_value > 0 else None for d in value_info.type.tensor_type.shape.dim],
    )


def construct_decimation_wrapper(
    model_a: onnx.ModelProto,
    model_b: onnx.ModelProto,
    decimation: int,
    opset_version: int,
    ir_version: int,
) -> onnx.ModelProto:
    """
    Wraps two ONNX models with decimation logic. Executes model_a if (step_count % decimation == 0), otherwise model_b.
    Args:
        model_a: ONNX submodel for decimation event.
        model_b: ONNX submodel for other steps.
        decimation: Decimation factor.
    Returns:
        An ONNX ModelProto with fixed periodic conditional branching.
    """
    time_input = onnx.helper.make_tensor_value_info("ctx.step_count", onnx.TensorProto.INT32, [])

    # Combine inputs and outputs from both models
    inputs = list(
        {i.name: _copy_value_info(i) for m in [model_a, model_b] for i in m.graph.input}.values()
    )
    outputs = list(
        {o.name: _copy_value_info(o) for m in [model_a, model_b] for o in m.graph.output}.values()
    )

    decimation_const = onnx.helper.make_tensor(
        "decimation", onnx.TensorProto.INT32, (), [decimation]
    )
    zero_const = onnx.helper.make_tensor("zero", onnx.TensorProto.INT32, (), [0])
    mod_node = onnx.helper.make_node("Mod", ["ctx.step_count", "decimation"], ["is_event"])
    eq_node = onnx.helper.make_node("Equal", ["is_event", "zero"], ["cond"])

    # Remove submodel inputs (will be passed by parent graph)
    for g in (model_a.graph, model_b.graph):
        del g.input[:]

    # Branching node
    if_node = onnx.helper.make_node(
        "If",
        inputs=["cond"],
        outputs=[o.name for o in outputs],
        then_branch=model_a.graph,
        else_branch=model_b.graph,
    )

    parent_graph = onnx.helper.make_graph(
        nodes=[mod_node, eq_node, if_node],
        name="decimation_wrapper",
        inputs=[time_input] + inputs,
        outputs=outputs,
        initializer=[decimation_const, zero_const],
    )

    model = onnx.helper.make_model(
        parent_graph,
        producer_name="construct_decimation_wrapper",
        opset_imports=[onnx.helper.make_operatorsetid("", opset_version)],
        ir_version=ir_version,
    )
    onnx.checker.check_model(model)
    return model
