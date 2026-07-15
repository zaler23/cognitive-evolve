"""Bounded typed JSON DSL for deterministic in-process Z3 checks."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from cognitive_evolve_runtime.core.serialization import stable_hash

from .types import VerificationResult


DSL_VERSION = "z3_dsl/v1"
MAX_SYMBOLS = 32
MAX_CONSTRAINTS = 64
MAX_AST_DEPTH = 16
MAX_AST_NODES = 1024
MAX_OPERATOR_ARGS = 32
MAX_BITVEC_WIDTH = 64
MAX_INTEGER_BITS = 256
Z3_TIMEOUT_MS = 1000

_SYMBOL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
_BOOL_OPERATORS = {"and", "or", "not", "implies", "xor"}
_RELATION_OPERATORS = {"eq", "ne", "distinct", "lt", "lte", "gt", "gte"}
_ARITHMETIC_OPERATORS = {"add", "sub", "mul", "neg"}
_BITVEC_OPERATORS = {"bvand", "bvor", "bvxor", "bvnot", "bvshl", "bvlshr"}
_OTHER_OPERATORS = {"ite"}
_OPERATORS = _BOOL_OPERATORS | _RELATION_OPERATORS | _ARITHMETIC_OPERATORS | _BITVEC_OPERATORS | _OTHER_OPERATORS


class _RejectedDsl(ValueError):
    pass


@dataclass(frozen=True)
class _Sort:
    kind: str
    width: int = 0


_BOOL = _Sort("Bool")
_INT = _Sort("Int")


def evaluate_z3_dsl(
    dsl: Any,
    *,
    expected_status: str,
    fingerprint: str = "",
    formal_kind: str = "satisfiability",
) -> VerificationResult:
    """Validate before import, solve under engine limits, and emit a verification receipt."""

    dsl_sha256 = stable_hash(dsl)
    try:
        symbol_sorts = _validate_dsl(dsl)
    except _RejectedDsl as exc:
        return _receipt(
            status="rejected",
            dsl_sha256=dsl_sha256,
            expected_status=expected_status,
            fingerprint=fingerprint,
            formal_kind=formal_kind,
            reason=str(exc),
        )

    try:
        import z3  # type: ignore
    except Exception as exc:  # External optional binding may be absent or unloadable.
        return _receipt(
            status="unavailable",
            dsl_sha256=dsl_sha256,
            expected_status=expected_status,
            fingerprint=fingerprint,
            formal_kind=formal_kind,
            reason="z3_solver_python_binding_unavailable:" + type(exc).__name__,
        )

    try:
        symbols = {
            name: z3.Bool(name)
            if sort == _BOOL
            else z3.Int(name)
            if sort == _INT
            else z3.BitVec(name, sort.width)
            for name, sort in symbol_sorts.items()
        }
        constraints = [_compile_expr(node, symbols, z3) for node in dsl["constraints"]]
        solver = z3.Solver()
        solver.set(timeout=Z3_TIMEOUT_MS)
        solver.add(*constraints)
        raw_status = solver.check()
        if raw_status == z3.sat:
            status = "sat"
            reason = ""
        elif raw_status == z3.unsat:
            status = "unsat"
            reason = ""
        else:
            reason = str(solver.reason_unknown() or "unknown")
            lowered = reason.lower()
            status = "timeout" if "timeout" in lowered or "canceled" in lowered else "unknown"
    except Exception as exc:  # Z3 is an external tool boundary; never terminate the evolution run.
        return _receipt(
            status="unavailable",
            dsl_sha256=dsl_sha256,
            expected_status=expected_status,
            fingerprint=fingerprint,
            formal_kind=formal_kind,
            reason="z3_solver_error:" + type(exc).__name__,
        )

    return _receipt(
        status=status,
        dsl_sha256=dsl_sha256,
        expected_status=expected_status,
        fingerprint=fingerprint,
        formal_kind=formal_kind,
        reason=reason,
    )


def _receipt(
    *,
    status: str,
    dsl_sha256: str,
    expected_status: str,
    fingerprint: str,
    formal_kind: str,
    reason: str,
) -> VerificationResult:
    passed = status == expected_status
    replayable = status in {"sat", "unsat"}
    validation_status = (
        "preliminary_passed"
        if passed
        else "preliminary_failed"
        if replayable
        else "inconclusive"
        if status in {"unknown", "timeout"}
        else "not_run"
    )
    diagnostics = [f"z3_status:{status}", f"formal_kind:{formal_kind}"]
    if reason:
        diagnostics.append("z3_reason:" + reason)
    metadata = {
        "fingerprint": fingerprint or "verifier-" + dsl_sha256[:16],
        "cli_not_attempted": True,
        "oracle_kind": "formal",
        "diagnostics_only": not replayable,
        "replay_verified": bool(passed and replayable),
        "validation_status": validation_status,
        "formal_kind": formal_kind,
        "z3_status": status,
        "z3_expected_status": expected_status,
        "z3_dsl_sha256": dsl_sha256,
        "z3_reason": reason,
        "z3_timeout_ms": Z3_TIMEOUT_MS,
        "z3_limits": {
            "max_symbols": MAX_SYMBOLS,
            "max_constraints": MAX_CONSTRAINTS,
            "max_ast_depth": MAX_AST_DEPTH,
            "max_ast_nodes": MAX_AST_NODES,
            "max_operator_args": MAX_OPERATOR_ARGS,
            "max_bitvec_width": MAX_BITVEC_WIDTH,
            "max_integer_bits": MAX_INTEGER_BITS,
        },
    }
    return VerificationResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        evidence_ref="evidence-z3-" + stable_hash({"dsl_sha256": dsl_sha256, "status": status})[:16],
        replayable=replayable,
        diagnostics=diagnostics,
        metadata=metadata,
    )


def _validate_dsl(dsl: Any) -> dict[str, _Sort]:
    if not isinstance(dsl, dict):
        raise _RejectedDsl("dsl_must_be_object")
    extra = sorted(str(key) for key in dsl if key not in {"version", "symbols", "constraints"})
    if extra:
        raise _RejectedDsl("unsupported_top_level_fields:" + ",".join(extra))
    version = dsl.get("version", DSL_VERSION)
    if version != DSL_VERSION:
        raise _RejectedDsl("unsupported_dsl_version")
    declarations = dsl.get("symbols")
    constraints = dsl.get("constraints")
    if not isinstance(declarations, list):
        raise _RejectedDsl("symbols_must_be_array")
    if len(declarations) > MAX_SYMBOLS:
        raise _RejectedDsl("symbol_count_exceeds_limit")
    if not isinstance(constraints, list):
        raise _RejectedDsl("constraints_must_be_array")
    if not constraints:
        raise _RejectedDsl("constraints_required")
    if len(constraints) > MAX_CONSTRAINTS:
        raise _RejectedDsl("constraint_count_exceeds_limit")

    symbols: dict[str, _Sort] = {}
    for declaration in declarations:
        name, sort = _validate_symbol_declaration(declaration)
        if name in symbols:
            raise _RejectedDsl("duplicate_symbol:" + name)
        symbols[name] = sort

    node_count = [0]
    for constraint in constraints:
        if _validate_expr(constraint, symbols, depth=1, node_count=node_count) != _BOOL:
            raise _RejectedDsl("constraint_must_have_bool_sort")
    return symbols


def _validate_symbol_declaration(declaration: Any) -> tuple[str, _Sort]:
    if not isinstance(declaration, dict):
        raise _RejectedDsl("symbol_declaration_must_be_object")
    name = declaration.get("name")
    kind = declaration.get("sort")
    if not isinstance(name, str) or not _SYMBOL_NAME.fullmatch(name):
        raise _RejectedDsl("invalid_symbol_name")
    if kind in {"Bool", "Int"}:
        if set(declaration) != {"name", "sort"}:
            raise _RejectedDsl("unsupported_symbol_fields:" + name)
        return name, _BOOL if kind == "Bool" else _INT
    if kind == "BitVec":
        if set(declaration) != {"name", "sort", "width"}:
            raise _RejectedDsl("bitvec_symbol_width_required:" + name)
        return name, _Sort("BitVec", _validate_width(declaration.get("width")))
    raise _RejectedDsl("unsupported_sort:" + str(kind))


def _validate_expr(node: Any, symbols: dict[str, _Sort], *, depth: int, node_count: list[int]) -> _Sort:
    if depth > MAX_AST_DEPTH:
        raise _RejectedDsl("ast_depth_exceeds_limit")
    node_count[0] += 1
    if node_count[0] > MAX_AST_NODES:
        raise _RejectedDsl("ast_node_count_exceeds_limit")
    if not isinstance(node, dict):
        raise _RejectedDsl("expression_node_must_be_object")
    op = node.get("op")
    if not isinstance(op, str):
        raise _RejectedDsl("operator_required")

    if op == "symbol":
        if set(node) != {"op", "name"}:
            raise _RejectedDsl("invalid_symbol_node")
        name = node.get("name")
        if not isinstance(name, str) or name not in symbols:
            raise _RejectedDsl("unknown_symbol:" + str(name))
        return symbols[name]
    if op == "bool":
        if set(node) != {"op", "value"} or not isinstance(node.get("value"), bool):
            raise _RejectedDsl("invalid_bool_literal")
        return _BOOL
    if op == "int":
        value = node.get("value")
        if set(node) != {"op", "value"} or isinstance(value, bool) or not isinstance(value, int):
            raise _RejectedDsl("invalid_int_literal")
        if value.bit_length() > MAX_INTEGER_BITS:
            raise _RejectedDsl("integer_literal_exceeds_limit")
        return _INT
    if op == "bitvec":
        value = node.get("value")
        if set(node) != {"op", "value", "width"} or isinstance(value, bool) or not isinstance(value, int):
            raise _RejectedDsl("invalid_bitvec_literal")
        width = _validate_width(node.get("width"))
        if value < 0 or value >= 1 << width:
            raise _RejectedDsl("bitvec_literal_out_of_range")
        return _Sort("BitVec", width)
    if op not in _OPERATORS:
        raise _RejectedDsl("unsupported_operator:" + op)
    if set(node) != {"op", "args"} or not isinstance(node.get("args"), list):
        raise _RejectedDsl("operator_args_must_be_array:" + op)
    args = node["args"]
    if len(args) > MAX_OPERATOR_ARGS:
        raise _RejectedDsl("operator_arg_count_exceeds_limit:" + op)
    sorts = [_validate_expr(arg, symbols, depth=depth + 1, node_count=node_count) for arg in args]
    return _validate_operator(op, sorts)


def _validate_operator(op: str, sorts: list[_Sort]) -> _Sort:
    if op == "not":
        _require_arity(op, sorts, 1)
        _require_all(op, sorts, _BOOL)
        return _BOOL
    if op in {"and", "or", "xor"}:
        _require_range(op, sorts, 2, MAX_OPERATOR_ARGS)
        _require_all(op, sorts, _BOOL)
        return _BOOL
    if op == "implies":
        _require_arity(op, sorts, 2)
        _require_all(op, sorts, _BOOL)
        return _BOOL
    if op in {"eq", "ne"}:
        _require_arity(op, sorts, 2)
        _require_same(op, sorts)
        return _BOOL
    if op == "distinct":
        _require_range(op, sorts, 2, MAX_OPERATOR_ARGS)
        _require_same(op, sorts)
        return _BOOL
    if op in {"lt", "lte", "gt", "gte"}:
        _require_arity(op, sorts, 2)
        _require_same(op, sorts)
        _require_numeric(op, sorts[0])
        return _BOOL
    if op in {"add", "mul"}:
        _require_range(op, sorts, 2, MAX_OPERATOR_ARGS)
        _require_same(op, sorts)
        _require_numeric(op, sorts[0])
        return sorts[0]
    if op == "sub":
        _require_arity(op, sorts, 2)
        _require_same(op, sorts)
        _require_numeric(op, sorts[0])
        return sorts[0]
    if op == "neg":
        _require_arity(op, sorts, 1)
        _require_numeric(op, sorts[0])
        return sorts[0]
    if op in {"bvand", "bvor", "bvxor", "bvshl", "bvlshr"}:
        _require_arity(op, sorts, 2)
        _require_same(op, sorts)
        _require_bitvec(op, sorts[0])
        return sorts[0]
    if op == "bvnot":
        _require_arity(op, sorts, 1)
        _require_bitvec(op, sorts[0])
        return sorts[0]
    _require_arity("ite", sorts, 3)
    if sorts[0] != _BOOL or sorts[1] != sorts[2]:
        raise _RejectedDsl("sort_mismatch:ite")
    return sorts[1]


def _compile_expr(node: dict[str, Any], symbols: dict[str, Any], z3: Any) -> Any:
    op = node["op"]
    if op == "symbol":
        return symbols[node["name"]]
    if op == "bool":
        return z3.BoolVal(node["value"])
    if op == "int":
        return z3.IntVal(node["value"])
    if op == "bitvec":
        return z3.BitVecVal(node["value"], node["width"])
    args = [_compile_expr(arg, symbols, z3) for arg in node["args"]]
    if op == "and":
        return z3.And(*args)
    if op == "or":
        return z3.Or(*args)
    if op == "not":
        return z3.Not(args[0])
    if op == "implies":
        return z3.Implies(args[0], args[1])
    if op == "xor":
        result = args[0]
        for arg in args[1:]:
            result = z3.Xor(result, arg)
        return result
    if op == "eq":
        return args[0] == args[1]
    if op == "ne":
        return z3.Not(args[0] == args[1])
    if op == "distinct":
        return z3.Distinct(*args)
    if op == "lt":
        return args[0] < args[1]
    if op == "lte":
        return args[0] <= args[1]
    if op == "gt":
        return args[0] > args[1]
    if op == "gte":
        return args[0] >= args[1]
    if op in {"add", "mul"}:
        result = args[0]
        for arg in args[1:]:
            result = result + arg if op == "add" else result * arg
        return result
    if op == "sub":
        return args[0] - args[1]
    if op == "neg":
        return -args[0]
    if op == "bvand":
        return args[0] & args[1]
    if op == "bvor":
        return args[0] | args[1]
    if op == "bvxor":
        return args[0] ^ args[1]
    if op == "bvnot":
        return ~args[0]
    if op == "bvshl":
        return args[0] << args[1]
    if op == "bvlshr":
        return z3.LShR(args[0], args[1])
    return z3.If(args[0], args[1], args[2])


def _validate_width(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _RejectedDsl("invalid_bitvec_width")
    if value > MAX_BITVEC_WIDTH:
        raise _RejectedDsl("bitvec_width_exceeds_limit")
    return value


def _require_arity(op: str, sorts: list[_Sort], expected: int) -> None:
    if len(sorts) != expected:
        raise _RejectedDsl(f"invalid_arity:{op}:expected_{expected}")


def _require_range(op: str, sorts: list[_Sort], minimum: int, maximum: int) -> None:
    if not minimum <= len(sorts) <= maximum:
        raise _RejectedDsl(f"invalid_arity:{op}:expected_{minimum}_to_{maximum}")


def _require_all(op: str, sorts: list[_Sort], expected: _Sort) -> None:
    if any(sort != expected for sort in sorts):
        raise _RejectedDsl("sort_mismatch:" + op)


def _require_same(op: str, sorts: list[_Sort]) -> None:
    if not sorts or any(sort != sorts[0] for sort in sorts[1:]):
        raise _RejectedDsl("sort_mismatch:" + op)


def _require_numeric(op: str, sort: _Sort) -> None:
    if sort.kind not in {"Int", "BitVec"}:
        raise _RejectedDsl("numeric_sort_required:" + op)


def _require_bitvec(op: str, sort: _Sort) -> None:
    if sort.kind != "BitVec":
        raise _RejectedDsl("bitvec_sort_required:" + op)


__all__ = [
    "DSL_VERSION",
    "MAX_AST_DEPTH",
    "MAX_AST_NODES",
    "MAX_BITVEC_WIDTH",
    "MAX_CONSTRAINTS",
    "MAX_INTEGER_BITS",
    "MAX_OPERATOR_ARGS",
    "MAX_SYMBOLS",
    "Z3_TIMEOUT_MS",
    "evaluate_z3_dsl",
]
