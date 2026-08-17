from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OP_ROOT = ROOT / "fla/ops/ascendc/kda/chunk_kda_bwd"


def _read(relative: str) -> str:
    return (OP_ROOT / relative).read_text(encoding="utf-8")


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def test_reserved_attrs_default_to_current_implementation():
    op_def = _read("op_host/chunk_kda_bwd_def.cpp")
    assert 'Attr("disable_recompute").AttrType(OPTIONAL).Bool(true)' in op_def
    assert 'Attr("use_exp2").AttrType(OPTIONAL).Bool(true)' in op_def


def test_aclnn_and_l0_keep_the_same_reserved_attr_order():
    aclnn_header = _read("op_host/op_api/aclnn_chunk_kda_bwd.h")
    l0_header = _read("op_host/op_api/chunk_kda_bwd.h")
    l0_source = _read("op_host/op_api/chunk_kda_bwd.cpp")
    signature = "bool disableRecompute, bool useExp2"
    assert signature in _normalize_whitespace(aclnn_header)
    assert signature in _normalize_whitespace(l0_header)
    assert (
        "OP_ATTR(scale, chunkSize, safeGate, useGateInKernel, lowerBound, "
        "disableRecompute, useExp2)"
    ) in _normalize_whitespace(l0_source)


def test_false_reserved_modes_fail_before_launch():
    aclnn = _read("op_host/op_api/aclnn_chunk_kda_bwd.cpp")
    tiling = _read("op_host/chunk_kda_bwd_tiling.cpp")
    for attr, message in (
        ("disableRecompute", "disable_recompute=false is reserved but not supported"),
        ("useExp2", "use_exp2=false is reserved but not supported"),
    ):
        assert f"CHECK_COND({attr}" in aclnn
        assert message in aclnn
        assert f"!*{attr}" in tiling
        assert message in tiling
