"""Exact arithmetic so the model never computes in its head.

Evaluates an expression through a filtered Python AST (numbers, + - * / // % **, parentheses,
a few math functions and constants) — never ``eval``. The model writes the expression in Python
syntax; the tool returns the raw number.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Dict, Optional

from ..base import Tool, ToolContext
from ..types import ToolExecutionResult

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "log": math.log, "log10": math.log10,
          "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan, "min": min, "max": max,
          "floor": math.floor, "ceil": math.ceil}
_CONSTS = {"pi": math.pi, "e": math.e}


class CalculationError(ValueError):
    pass


def evaluate(expression: str) -> float | int:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            left, right = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(right) > 100:
                raise CalculationError("exponent too large")
            return _OPS[type(n.op)](left, right)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name) and n.id in _CONSTS:
            return _CONSTS[n.id]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(x) for x in n.args])
        raise CalculationError("unsupported expression")
    try:
        value = ev(ast.parse(expression.replace("^", "**"), mode="eval"))
    except ZeroDivisionError:
        raise CalculationError("division by zero") from None
    except CalculationError:
        raise
    except Exception:
        raise CalculationError("could not parse expression") from None
    return round(value, 10) if isinstance(value, float) else value


class CalculateTool(Tool):
    @property
    def name(self) -> str:
        return "calculate"

    @property
    def description(self) -> str:
        return ("Compute an arithmetic expression exactly (percentages, conversions, anything beyond one "
                "trivial operation). Write it in Python syntax, e.g. 0.15 * 87 or sqrt(144) + 2**3.")

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        expr = str((args or {}).get("expression", "")).strip()
        try:
            return ToolExecutionResult(True, f"{expr} = {evaluate(expr)}")
        except CalculationError as e:
            return ToolExecutionResult(False, None, error_message=f"Cannot compute '{expr}': {e}")
