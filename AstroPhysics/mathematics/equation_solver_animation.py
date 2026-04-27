#!/usr/bin/env python3
"""
Centered algebra animation for simple one-variable linear equations.

Visual design:
- black background
- equation centered on screen
- slow, eased token motion
- no graphing

Interaction:
- no terminal input after launch
- press Enter in the window to solve the typed equation
- press n to edit a new equation inside the window

Supported equation form:
- one variable: x
- linear only
- examples:
  - 2*x + 3 = 11
  - 5*x - 7 = 2*x + 8
  - x/3 + 5 = 9
  - -2*x + 4 = 10
"""

from __future__ import annotations

import ast
import math
import os
import re
from math import isqrt
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WINDOW_NAME = "Equation Solver Animation"
WIDTH = 1400
HEIGHT = 860
BG = (0, 0, 0)
TEXT = (255, 255, 255)
MUTED = (215, 215, 215)
ACCENT = (255, 255, 255)
STEP_COLOR = (90, 90, 255)
ERROR = (255, 120, 120)
INPUT_BORDER = (255, 255, 255)
PANEL = (16, 16, 16)
PANEL_EDGE = (58, 58, 58)
GRID = (54, 54, 54)
SOFT = (150, 150, 150)
FONT = cv2.FONT_HERSHEY_DUPLEX
TITLE_SCALE = 0.95
INFO_SCALE = 0.62
EQUATION_SCALE = 1.8
THICKNESS = 2
TOKEN_GAP = 22
DEFAULT_FRAMES_PER_STEP = 120
DEFAULT_HOLD_FRAMES = 18
MIN_FRAMES_PER_STEP = 24
MAX_FRAMES_PER_STEP = 280
SPEED_STEP = 12
DEFAULT_EQUATION = ""
FRAME_DELAY_MS = 16
INTEGRAL_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/STIXIntUpDBol.otf",
    "/System/Library/Fonts/Supplemental/STIXIntUpBol.otf",
    "/System/Library/Fonts/Supplemental/STIXGeneralBol.otf",
    "/System/Library/Fonts/Supplemental/STIXTwoMath.otf",
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
]


class TextSprite(NamedTuple):
    image: np.ndarray
    alpha: np.ndarray
    baseline: int
    width: int
    height: int


TEXT_CACHE: dict[tuple[str, tuple[int, int, int], float], TextSprite] = {}
INTEGRAL_FONT_PATH: str | None = None


@dataclass
class LinearExpr:
    coef: Fraction
    const: Fraction


@dataclass
class QuadraticExpr:
    quad: Fraction
    linear: Fraction
    const: Fraction


@dataclass
class SolveStep:
    label: str
    tokens: list[str]


@dataclass
class GraphCurve:
    expression_text: str
    y_values: list[float | None]
    color: tuple[int, int, int]


@dataclass
class GraphSpec:
    expressions_text: str
    x_values: list[float]
    curves: list[GraphCurve]
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass
class ProblemSpec:
    steps: list[SolveStep]
    display_text: str
    formula_text: str
    graph_spec: GraphSpec | None = None


@dataclass(frozen=True)
class SymbolicTerm:
    basis: str
    coeff: Fraction
    power: int | None = None


def clean_number(value: float) -> str:
    frac = value if isinstance(value, Fraction) else Fraction(str(value))
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"


def clean_decimal(value: float, digits: int = 6) -> str:
    if not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if abs(value - round(value)) < 10 ** (-digits):
        return str(int(round(value)))
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def is_zero(value: Fraction) -> bool:
    return value == 0


def is_one(value: Fraction) -> bool:
    return value == 1


def format_term(coef: Fraction) -> str:
    if is_one(coef):
        return "x"
    if coef == -1:
        return "-x"
    if coef.denominator != 1 and abs(coef.numerator) == 1:
        sign = "-" if coef.numerator < 0 else ""
        return f"{sign}x/{coef.denominator}"
    if coef.denominator != 1:
        return f"{coef.numerator}x/{coef.denominator}"
    return f"{coef.numerator}x"


def normalize_input_text(text: str) -> str:
    normalized = text.replace("^", "**").strip()
    # Accept common handwritten shorthand such as 2x, 3(x+1), x(2+1), and 2sin(x).
    patterns = [
        (r"(?<=\d)(?=[A-Za-z(])", "*"),
        (r"(?<=[x\)])(?=\d)", "*"),
        (r"(?<=[x\)])(?=[A-Za-z(])", "*"),
    ]
    for pattern, replacement in patterns:
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def format_side(expr: LinearExpr) -> list[str]:
    tokens: list[str] = []
    has_x = not is_zero(expr.coef)
    has_const = not is_zero(expr.const)

    if has_x:
        tokens.append(format_term(expr.coef))

    if has_const:
        const_text = clean_number(abs(expr.const))
        if tokens:
            tokens.append("+" if expr.const > 0 else "-")
            tokens.append(const_text)
        else:
            tokens.append(clean_number(expr.const))

    if not tokens:
        tokens.append("0")

    return tokens


def format_equation(left: LinearExpr, right: LinearExpr) -> list[str]:
    return format_side(left) + ["="] + format_side(right)


def parse_linear(node: ast.AST) -> LinearExpr:
    if isinstance(node, ast.BinOp):
        left = parse_linear(node.left)
        right = parse_linear(node.right)

        if isinstance(node.op, ast.Add):
            return LinearExpr(left.coef + right.coef, left.const + right.const)
        if isinstance(node.op, ast.Sub):
            return LinearExpr(left.coef - right.coef, left.const - right.const)
        if isinstance(node.op, ast.Mult):
            if not is_zero(left.coef) and not is_zero(right.coef):
                raise ValueError("Nonlinear multiplication is not supported.")
            if is_zero(left.coef):
                return LinearExpr(left.const * right.coef, left.const * right.const)
            return LinearExpr(right.const * left.coef, right.const * left.const)
        if isinstance(node.op, ast.Div):
            if not is_zero(right.coef):
                raise ValueError("Division by a variable expression is not supported.")
            if is_zero(right.const):
                raise ValueError("Division by zero is not allowed.")
            return LinearExpr(left.coef / right.const, left.const / right.const)
        raise ValueError("Only +, -, *, and / are supported.")

    if isinstance(node, ast.UnaryOp):
        value = parse_linear(node.operand)
        if isinstance(node.op, ast.USub):
            return LinearExpr(-value.coef, -value.const)
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError("Unsupported unary operator.")

    if isinstance(node, ast.Name):
        if node.id != "x":
            raise ValueError("Only the variable x is supported.")
        return LinearExpr(Fraction(1), Fraction(0))

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return LinearExpr(Fraction(0), Fraction(str(node.value)))

    raise ValueError("Unsupported equation syntax.")


# ================= Quadratic equation parsing and solving =================

def parse_quadratic(node: ast.AST) -> QuadraticExpr:
    if isinstance(node, ast.BinOp):
        left = parse_quadratic(node.left)
        right = parse_quadratic(node.right)

        if isinstance(node.op, ast.Add):
            return QuadraticExpr(left.quad + right.quad, left.linear + right.linear, left.const + right.const)
        if isinstance(node.op, ast.Sub):
            return QuadraticExpr(left.quad - right.quad, left.linear - right.linear, left.const - right.const)
        if isinstance(node.op, ast.Mult):
            left_degree = quadratic_degree(left)
            right_degree = quadratic_degree(right)
            if left_degree + right_degree > 2:
                raise ValueError("Only quadratic equations up to x^2 are supported.")
            return multiply_quadratic(left, right)
        if isinstance(node.op, ast.Div):
            if right.quad != 0 or right.linear != 0:
                raise ValueError("Division by a variable expression is not supported.")
            if right.const == 0:
                raise ValueError("Division by zero is not allowed.")
            return QuadraticExpr(left.quad / right.const, left.linear / right.const, left.const / right.const)
        if isinstance(node.op, ast.Pow):
            base = parse_quadratic(node.left)
            if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, int):
                raise ValueError("Only integer powers are supported in quadratic equations.")
            power = int(node.right.value)
            if power < 0 or power > 2:
                raise ValueError("Only powers 0, 1, and 2 are supported in quadratic equations.")
            if power == 0:
                return QuadraticExpr(Fraction(0), Fraction(0), Fraction(1))
            if power == 1:
                return base
            squared = multiply_quadratic(base, base)
            if quadratic_degree(squared) > 2:
                raise ValueError("Only quadratic equations up to x^2 are supported.")
            return squared
        raise ValueError("Only +, -, *, /, and powers up to x^2 are supported.")

    if isinstance(node, ast.UnaryOp):
        value = parse_quadratic(node.operand)
        if isinstance(node.op, ast.USub):
            return QuadraticExpr(-value.quad, -value.linear, -value.const)
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError("Unsupported unary operator.")

    if isinstance(node, ast.Name):
        if node.id != "x":
            raise ValueError("Only the variable x is supported.")
        return QuadraticExpr(Fraction(0), Fraction(1), Fraction(0))

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return QuadraticExpr(Fraction(0), Fraction(0), Fraction(str(node.value)))

    raise ValueError("Unsupported quadratic equation syntax.")


def quadratic_degree(expr: QuadraticExpr) -> int:
    if expr.quad != 0:
        return 2
    if expr.linear != 0:
        return 1
    return 0


def multiply_quadratic(left: QuadraticExpr, right: QuadraticExpr) -> QuadraticExpr:
    quad = left.quad * right.const + left.linear * right.linear + left.const * right.quad
    linear = left.linear * right.const + left.const * right.linear
    const = left.const * right.const
    return QuadraticExpr(quad, linear, const)


def parse_quadratic_equation(text: str) -> tuple[QuadraticExpr, QuadraticExpr]:
    if "=" not in text:
        raise ValueError("Enter an equation with '='.")
    left_text, right_text = text.split("=", 1)
    left = parse_quadratic(ast.parse(left_text.strip(), mode="eval").body)
    right = parse_quadratic(ast.parse(right_text.strip(), mode="eval").body)
    return left, right


def subtract_quadratic(left: QuadraticExpr, right: QuadraticExpr) -> QuadraticExpr:
    return QuadraticExpr(left.quad - right.quad, left.linear - right.linear, left.const - right.const)


def format_quadratic_term(coef: Fraction, body: str) -> str:
    if coef == 1:
        return body
    if coef == -1:
        return f"-{body}"
    if coef.denominator != 1:
        return f"{coef.numerator}{body}/{coef.denominator}"
    return f"{coef.numerator}{body}"


def format_quadratic_expr(expr: QuadraticExpr) -> list[str]:
    pieces: list[tuple[Fraction, str]] = []
    if expr.quad != 0:
        pieces.append((expr.quad, "x^2"))
    if expr.linear != 0:
        pieces.append((expr.linear, "x"))
    if expr.const != 0:
        pieces.append((expr.const, ""))
    if not pieces:
        return ["0"]

    tokens: list[str] = []
    for idx, (coef, body) in enumerate(pieces):
        if idx == 0:
            tokens.append(clean_number(coef) if body == "" else format_quadratic_term(coef, body))
        else:
            tokens.append("-" if coef < 0 else "+")
            abs_coef = abs(coef)
            tokens.append(clean_number(abs_coef) if body == "" else format_quadratic_term(abs_coef, body))
    return tokens


def format_quadratic_equation(left: QuadraticExpr, right: QuadraticExpr) -> list[str]:
    return format_quadratic_expr(left) + ["="] + format_quadratic_expr(right)


def sqrt_fraction_exact(value: Fraction) -> Fraction | None:
    if value < 0:
        return None
    num_root = isqrt(value.numerator)
    den_root = isqrt(value.denominator)
    if num_root * num_root == value.numerator and den_root * den_root == value.denominator:
        return Fraction(num_root, den_root)
    return None


def find_integer_factor_pair(a: Fraction, b: Fraction, c: Fraction) -> tuple[int, int] | None:
    if a != 1 or b.denominator != 1 or c.denominator != 1:
        return None
    b_int = int(b)
    c_int = int(c)
    limit = abs(c_int) + 1
    for p in range(-limit, limit + 1):
        for q in range(-limit, limit + 1):
            if p + q == b_int and p * q == c_int:
                return p, q
    return None


def factor_token(value: int) -> str:
    if value == 0:
        return "x"
    if value > 0:
        return f"(x+{value})"
    return f"(x{value})"


def build_quadratic_steps(equation: str) -> list[SolveStep] | None:
    left, right = parse_quadratic_equation(equation)
    standard = subtract_quadratic(left, right)
    if standard.quad == 0:
        return None

    steps = [SolveStep("Starting quadratic equation", format_quadratic_equation(left, right))]
    steps.append(SolveStep("Move everything to one side", format_quadratic_equation(standard, QuadraticExpr(Fraction(0), Fraction(0), Fraction(0)))))

    a = standard.quad
    b = standard.linear
    c = standard.const

    if a != 0 and a != 1:
        steps.append(SolveStep("Identify coefficients", ["a", "=", clean_number(a), "b", "=", clean_number(b), "c", "=", clean_number(c)]))
    else:
        steps.append(SolveStep("Identify coefficients", ["a", "=", clean_number(a), "b", "=", clean_number(b), "c", "=", clean_number(c)]))

    factor_pair = find_integer_factor_pair(a, b, c)
    if factor_pair is not None:
        p, q = factor_pair
        steps.append(SolveStep("Find two numbers", [str(p), "and", str(q), "multiply", "to", clean_number(c), "and", "add", "to", clean_number(b)]))
        steps.append(SolveStep("Factor the quadratic", [factor_token(p), factor_token(q), "=", "0"]))
        steps.append(SolveStep("Use the zero product property", [factor_token(p), "=", "0", "or", factor_token(q), "=", "0"]))
        roots = [Fraction(-p), Fraction(-q)]
        if roots[0] == roots[1]:
            steps.append(SolveStep("Solution", ["x", "=", clean_number(roots[0])]))
        else:
            steps.append(SolveStep("Solutions", ["x", "=", clean_number(roots[0]), "or", "x", "=", clean_number(roots[1])]))
        return steps

    discriminant = b * b - 4 * a * c
    steps.append(SolveStep("Use the quadratic formula", ["x", "=", "(-b", "+-", "sqrt(b^2-4ac))", "/", "2a"]))
    steps.append(SolveStep("Compute the discriminant", ["D", "=", clean_number(b), "^2", "-", "4", "(", clean_number(a), ")", "(", clean_number(c), ")"]))
    steps.append(SolveStep("Simplify the discriminant", ["D", "=", clean_number(discriminant)]))

    if discriminant < 0:
        steps.append(SolveStep("No real solutions", ["D", "<", "0"]))
        steps.append(SolveStep("Solutions", ["complex", "roots"]))
        return steps

    sqrt_disc = sqrt_fraction_exact(discriminant)
    denominator = 2 * a
    if sqrt_disc is not None:
        root_one = (-b + sqrt_disc) / denominator
        root_two = (-b - sqrt_disc) / denominator
        steps.append(SolveStep("Take the square root", ["sqrt(D)", "=", clean_number(sqrt_disc)]))
        if root_one == root_two:
            steps.append(SolveStep("Solution", ["x", "=", clean_number(root_one)]))
        else:
            steps.append(SolveStep("Solutions", ["x", "=", clean_number(root_one), "or", "x", "=", clean_number(root_two)]))
        return steps

    numerator_left = f"{-b}+sqrt({clean_number(discriminant)})"
    numerator_right = f"{-b}-sqrt({clean_number(discriminant)})"
    steps.append(SolveStep("Solutions", ["x", "=", numerator_left, "/", clean_number(denominator), "or", "x", "=", numerator_right, "/", clean_number(denominator)]))
    return steps


def parse_equation(text: str) -> tuple[LinearExpr, LinearExpr]:
    if "=" not in text:
        raise ValueError("Enter an equation with '='.")
    left_text, right_text = text.split("=", 1)
    left = parse_linear(ast.parse(left_text.strip(), mode="eval").body)
    right = parse_linear(ast.parse(right_text.strip(), mode="eval").body)
    return left, right


def const_terms(value: Fraction) -> list[SymbolicTerm]:
    return [] if value == 0 else [SymbolicTerm("const", value)]


def is_constant_terms(terms: list[SymbolicTerm]) -> bool:
    return all(term.basis == "const" for term in terms)


def constant_value(terms: list[SymbolicTerm]) -> Fraction:
    if not is_constant_terms(terms):
        raise ValueError("Expected a constant-only expression.")
    return sum((term.coeff for term in terms), Fraction(0))


def scale_terms(terms: list[SymbolicTerm], factor: Fraction) -> list[SymbolicTerm]:
    return [SymbolicTerm(term.basis, term.coeff * factor, term.power) for term in terms]


def canonicalize_terms(terms: list[SymbolicTerm]) -> list[SymbolicTerm]:
    combined: dict[tuple[str, int | None], Fraction] = {}
    for term in terms:
        key = (term.basis, term.power)
        combined[key] = combined.get(key, Fraction(0)) + term.coeff

    ordered: list[SymbolicTerm] = []
    for (basis, power), coeff in combined.items():
        if coeff == 0:
            continue
        ordered.append(SymbolicTerm(basis, coeff, power))

    def order_key(term: SymbolicTerm) -> tuple[int, int]:
        if term.basis == "pow":
            return (0, -(term.power or 0))
        if term.basis == "sin":
            return (1, 0)
        if term.basis == "cos":
            return (2, 0)
        if term.basis == "exp":
            return (3, 0)
        return (4, 0)

    ordered.sort(key=order_key)
    return ordered


def parse_symbolic_terms(node: ast.AST) -> list[SymbolicTerm]:
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            return parse_symbolic_terms(node.left) + parse_symbolic_terms(node.right)
        if isinstance(node.op, ast.Sub):
            return parse_symbolic_terms(node.left) + scale_terms(parse_symbolic_terms(node.right), Fraction(-1))
        if isinstance(node.op, ast.Mult):
            left_terms = parse_symbolic_terms(node.left)
            right_terms = parse_symbolic_terms(node.right)
            if is_constant_terms(left_terms):
                return scale_terms(right_terms, constant_value(left_terms))
            if is_constant_terms(right_terms):
                return scale_terms(left_terms, constant_value(right_terms))
            raise ValueError("Only constant scaling is supported in derivatives/integrals.")
        if isinstance(node.op, ast.Div):
            numerator_terms = parse_symbolic_terms(node.left)
            denominator_terms = parse_symbolic_terms(node.right)
            if not is_constant_terms(denominator_terms):
                raise ValueError("Only division by a constant is supported.")
            divisor = constant_value(denominator_terms)
            if divisor == 0:
                raise ValueError("Division by zero is not allowed.")
            return scale_terms(numerator_terms, Fraction(1, 1) / divisor)
        if isinstance(node.op, ast.Pow):
            if not isinstance(node.left, ast.Name) or node.left.id != "x":
                raise ValueError("Only powers of x are supported.")
            if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, int):
                raise ValueError("Only integer powers of x are supported.")
            return [SymbolicTerm("pow", Fraction(1), int(node.right.value))]
        raise ValueError("Unsupported operator in expression.")

    if isinstance(node, ast.UnaryOp):
        value = parse_symbolic_terms(node.operand)
        if isinstance(node.op, ast.USub):
            return scale_terms(value, Fraction(-1))
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError("Unsupported unary operator.")

    if isinstance(node, ast.Name):
        if node.id != "x":
            raise ValueError("Only the variable x is supported.")
        return [SymbolicTerm("pow", Fraction(1), 1)]

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return [SymbolicTerm("const", Fraction(str(node.value)))]

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1:
        func_name = node.func.id
        arg = node.args[0]
        if not isinstance(arg, ast.Name) or arg.id != "x":
            raise ValueError("Only functions of x are supported.")
        if func_name in ("sin", "cos", "exp"):
            return [SymbolicTerm(func_name, Fraction(1))]
        raise ValueError("Supported functions are sin(x), cos(x), and exp(x).")

    raise ValueError("Unsupported expression syntax.")


def parse_expression_terms(text: str) -> list[SymbolicTerm]:
    parsed = ast.parse(text.strip(), mode="eval").body
    return canonicalize_terms(parse_symbolic_terms(parsed))


def abs_term(term: SymbolicTerm) -> SymbolicTerm:
    return SymbolicTerm(term.basis, abs(term.coeff), term.power)


def term_body_text(term: SymbolicTerm) -> str:
    if term.basis == "const":
        return ""
    if term.basis == "pow":
        power = term.power or 1
        return "x" if power == 1 else f"x^{power}"
    if term.basis == "sin":
        return "sin(x)"
    if term.basis == "cos":
        return "cos(x)"
    if term.basis == "exp":
        return "exp(x)"
    raise ValueError("Unknown symbolic term.")


def format_symbolic_term(term: SymbolicTerm) -> str:
    if term.basis == "const":
        return clean_number(term.coeff)

    body = term_body_text(term)
    coeff = term.coeff
    if coeff == 1:
        return body
    if coeff == -1:
        return f"-{body}"

    sign = "-" if coeff < 0 else ""
    abs_coeff = abs(coeff)
    if abs_coeff.denominator != 1:
        if abs_coeff.numerator == 1:
            return f"{sign}{body}/{abs_coeff.denominator}"
        return f"{sign}{abs_coeff.numerator}{body}/{abs_coeff.denominator}"
    return f"{sign}{abs_coeff.numerator}{body}"


def symbolic_terms_to_tokens(terms: list[SymbolicTerm], keep_zero_terms: bool = False) -> list[str]:
    prepared = terms if keep_zero_terms else canonicalize_terms(terms)
    if not prepared:
        return ["0"]

    tokens: list[str] = []
    for idx, term in enumerate(prepared):
        if term.coeff == 0 and not keep_zero_terms:
            continue
        if idx == 0:
            tokens.append(format_symbolic_term(term))
        else:
            if term.coeff < 0:
                tokens.append("-")
                tokens.append(format_symbolic_term(abs_term(term)))
            else:
                tokens.append("+")
                tokens.append(format_symbolic_term(term))
    return tokens or ["0"]


def op_tokens_for_terms(prefix: str, terms: list[SymbolicTerm], suffix: list[str] | None = None) -> list[str]:
    suffix = suffix or []
    tokens: list[str] = []
    for idx, term in enumerate(terms):
        sign = "-" if term.coeff < 0 else "+"
        inner = symbolic_terms_to_tokens([abs_term(term)])
        if idx == 0:
            if sign == "-":
                tokens.append("-")
        else:
            tokens.append(sign)
        tokens.extend([prefix, "("])
        tokens.extend(inner)
        tokens.append(")")
        tokens.extend(suffix)
    return tokens or [prefix, "(", "0", ")"] + suffix


def derivative_of_term(term: SymbolicTerm) -> list[SymbolicTerm]:
    coeff = term.coeff
    if term.basis == "const":
        return [SymbolicTerm("const", Fraction(0))]
    if term.basis == "pow":
        power = term.power or 1
        if power == 0:
            return [SymbolicTerm("const", Fraction(0))]
        new_coeff = coeff * power
        if power - 1 == 0:
            return [SymbolicTerm("const", new_coeff)]
        return [SymbolicTerm("pow", new_coeff, power - 1)]
    if term.basis == "sin":
        return [SymbolicTerm("cos", coeff)]
    if term.basis == "cos":
        return [SymbolicTerm("sin", -coeff)]
    if term.basis == "exp":
        return [SymbolicTerm("exp", coeff)]
    raise ValueError("Unsupported derivative rule.")


def integral_of_term(term: SymbolicTerm) -> list[SymbolicTerm]:
    coeff = term.coeff
    if term.basis == "const":
        return [SymbolicTerm("pow", coeff, 1)]
    if term.basis == "pow":
        power = term.power or 1
        if power == -1:
            raise ValueError("Integral of 1/x is not supported in this animator.")
        return [SymbolicTerm("pow", coeff / Fraction(power + 1), power + 1)]
    if term.basis == "sin":
        return [SymbolicTerm("cos", -coeff)]
    if term.basis == "cos":
        return [SymbolicTerm("sin", coeff)]
    if term.basis == "exp":
        return [SymbolicTerm("exp", coeff)]
    raise ValueError("Unsupported integral rule.")


def build_derivative_steps(expression_text: str) -> list[SolveStep]:
    terms = parse_expression_terms(expression_text)
    steps = [SolveStep("Starting derivative", ["d/dx", "("] + symbolic_terms_to_tokens(terms) + [")"])]

    if len(terms) > 1:
        steps.append(SolveStep("Differentiate term by term", op_tokens_for_terms("d/dx", terms)))

    raw_result: list[SymbolicTerm] = []
    solved_groups: list[list[str]] = []
    for idx, term in enumerate(terms):
        raw_group = derivative_rule_tokens(term)
        steps.append(
            SolveStep(
                f"Apply derivative rule to {format_symbolic_term(abs_term(term))}",
                progressive_term_expression(terms, idx, solved_groups, raw_group, derivative_piece_tokens),
            )
        )
        simplified_group = symbolic_terms_to_tokens(derivative_of_term(abs_term(term)))
        if simplified_group != raw_group:
            steps.append(
                SolveStep(
                    f"Simplify derivative of {format_symbolic_term(abs_term(term))}",
                    progressive_term_expression(terms, idx, solved_groups, simplified_group, derivative_piece_tokens),
                )
            )
        solved_groups.append(simplified_group)
        raw_result.extend(derivative_of_term(term))

    combined_raw = join_signed_groups(solved_groups, [-1 if term.coeff < 0 else 1 for term in terms])
    if combined_raw != steps[-1].tokens:
        steps.append(SolveStep("Collect derivative terms", combined_raw))

    simplified = canonicalize_terms(raw_result)
    if symbolic_terms_to_tokens(simplified) != combined_raw:
        steps.append(SolveStep("Simplify", symbolic_terms_to_tokens(simplified)))
    steps.append(SolveStep("Derivative", symbolic_terms_to_tokens(simplified)))
    return steps


def integral_token(lower: str | None = None, upper: str | None = None) -> str:
    return f"INT|{lower or ''}|{upper or ''}"


def resolve_integral_font_path() -> str | None:
    global INTEGRAL_FONT_PATH
    if INTEGRAL_FONT_PATH is not None:
        return INTEGRAL_FONT_PATH
    for path in INTEGRAL_FONT_CANDIDATES:
        if os.path.exists(path):
            INTEGRAL_FONT_PATH = path
            return path
    INTEGRAL_FONT_PATH = ""
    return None


def parse_constant_fraction(text: str) -> Fraction:
    def walk(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp):
            value = walk(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
        if isinstance(node, ast.BinOp):
            left = walk(node.left)
            right = walk(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("Integral bound division by zero is not allowed.")
                return left / right
        raise ValueError("Integral bounds must be numeric constants.")

    return walk(ast.parse(text.strip(), mode="eval").body)


def parse_numeric_expression(text: str) -> ast.AST:
    return ast.parse(text.strip(), mode="eval").body


def evaluate_numeric_node(node: ast.AST, variables: dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in variables:
            return float(variables[node.id])
        if node.id == "pi":
            return math.pi
        if node.id == "e":
            return math.e
        raise ValueError(f"Unknown variable {node.id}.")
    if isinstance(node, ast.UnaryOp):
        value = evaluate_numeric_node(node.operand, variables)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError("Unsupported unary operator.")
    if isinstance(node, ast.BinOp):
        left = evaluate_numeric_node(node.left, variables)
        right = evaluate_numeric_node(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise ValueError("Unsupported numeric operator.")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1:
        arg = evaluate_numeric_node(node.args[0], variables)
        funcs = {
            "sin": math.sin,
            "cos": math.cos,
            "exp": math.exp,
            "abs": abs,
        }
        if node.func.id not in funcs:
            raise ValueError(f"Unsupported function {node.func.id}.")
        return float(funcs[node.func.id](arg))
    raise ValueError("Unsupported numeric expression.")


def evaluate_numeric_expression(node: ast.AST, variable_name: str, variable_value: float) -> float:
    return evaluate_numeric_node(node, {variable_name: variable_value})


def parse_limit_request(text: str) -> tuple[str, str, str]:
    match = re.match(r"^(?:lim|limit)\[(\w+)\s*->\s*([^\]]+)\]\s+(.+)$", text.strip(), re.IGNORECASE)
    if not match:
        raise ValueError("Limit input must look like lim[x->0] sin(x)/x.")
    return match.group(1).strip(), match.group(2).strip(), match.group(3).strip()


def parse_sum_request(text: str) -> tuple[str, int, int, str]:
    match = re.match(r"^(?:sum|sigma)\[(\w+)\s*=\s*([^,]+)\s*,\s*([^\]]+)\]\s+(.+)$", text.strip(), re.IGNORECASE)
    if not match:
        raise ValueError("Sum input must look like sum[i=1,4] i^2.")
    var_name = match.group(1).strip()
    start = parse_constant_fraction(match.group(2).strip())
    end = parse_constant_fraction(match.group(3).strip())
    if start.denominator != 1 or end.denominator != 1:
        raise ValueError("Sigma bounds must be integers.")
    return var_name, int(start), int(end), match.group(4).strip()


def parse_graph_request(text: str) -> str:
    match = re.match(r"^(?:graph|plot)\s+(.+)$", text.strip(), re.IGNORECASE)
    if not match:
        raise ValueError("Graph input must start with graph or plot.")
    expression = match.group(1).strip()
    if expression.lower().startswith("y="):
        expression = expression[2:].strip()
    elif re.match(r"^y\s*=", expression, re.IGNORECASE):
        expression = re.sub(r"^y\s*=", "", expression, count=1, flags=re.IGNORECASE).strip()
    return expression


def strip_graph_prefix(expression: str) -> str:
    expression = expression.strip()
    if expression.lower().startswith("y="):
        return expression[2:].strip()
    if re.match(r"^y\s*=", expression, re.IGNORECASE):
        return re.sub(r"^y\s*=", "", expression, count=1, flags=re.IGNORECASE).strip()
    return expression


def split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in text:
        if ch in "([{" :
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            piece = "".join(current).strip()
            if piece:
                parts.append(piece)
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def evaluate_term_at(term: SymbolicTerm, x_value: Fraction) -> Fraction:
    if term.basis == "const":
        return term.coeff
    if term.basis == "pow":
        return term.coeff * (x_value ** (term.power or 1))
    raise ValueError("Definite integral evaluation currently supports polynomial terms only.")


def substitution_tokens(terms: list[SymbolicTerm], x_value: Fraction) -> list[str]:
    tokens: list[str] = []
    for idx, term in enumerate(terms):
        if term.coeff == 0:
            continue
        value_text = clean_number(evaluate_term_at(term, x_value))
        if idx == 0:
            tokens.append(value_text)
        else:
            if value_text.startswith("-"):
                tokens.append("-")
                tokens.append(value_text[1:])
            else:
                tokens.append("+")
                tokens.append(value_text)
    return tokens or ["0"]


def value_tokens(terms: list[SymbolicTerm], x_value: Fraction) -> list[str]:
    total = sum((evaluate_term_at(term, x_value) for term in terms), Fraction(0))
    return [clean_number(total)]


def join_signed_groups(groups: list[list[str]], signs: list[int]) -> list[str]:
    tokens: list[str] = []
    for idx, (group, sign) in enumerate(zip(groups, signs)):
        if idx == 0:
            if sign < 0:
                tokens.append("-")
        else:
            tokens.append("-" if sign < 0 else "+")
        tokens.extend(group)
    return tokens or ["0"]


def derivative_piece_tokens(term: SymbolicTerm) -> list[str]:
    return ["d/dx", "("] + symbolic_terms_to_tokens([abs_term(term)]) + [")"]


def integral_piece_tokens(term: SymbolicTerm, int_tok: str) -> list[str]:
    return [int_tok, "("] + symbolic_terms_to_tokens([abs_term(term)]) + [")", "dx"]


def progressive_term_expression(
    terms: list[SymbolicTerm],
    current_index: int,
    solved_groups: list[list[str]],
    current_group: list[str],
    pending_builder,
) -> list[str]:
    groups: list[list[str]] = []
    for idx, term in enumerate(terms):
        if idx < current_index:
            groups.append(solved_groups[idx])
        elif idx == current_index:
            groups.append(current_group)
        else:
            groups.append(pending_builder(term))
    return join_signed_groups(groups, [-1 if term.coeff < 0 else 1 for term in terms])


def derivative_rule_tokens(term: SymbolicTerm) -> list[str]:
    item = abs_term(term)
    if item.basis == "const":
        return ["0"]
    if item.basis == "pow":
        power = item.power or 1
        factor = item.coeff * power
        if power == 0:
            return ["0"]
        if power == 1:
            return [clean_number(factor)]
        prefix = "" if factor == 1 else clean_number(factor)
        return [f"{prefix}x^({power}-1)"]
    return symbolic_terms_to_tokens(derivative_of_term(item))


def integral_rule_tokens(term: SymbolicTerm) -> list[str]:
    item = abs_term(term)
    if item.basis == "const":
        return [f"{clean_number(item.coeff)}x" if item.coeff != 1 else "x"]
    if item.basis == "pow":
        power = item.power or 1
        prefix = "" if item.coeff == 1 else clean_number(item.coeff)
        return [f"{prefix}x^({power}+1)/({power}+1)"]
    return symbolic_terms_to_tokens(integral_of_term(item))


def antiderivative_rule_tokens(term: SymbolicTerm) -> list[str]:
    coeff = term.coeff
    if term.basis == "pow":
        power = term.power or 1
        if power == -1:
            raise ValueError("Integral of 1/x is not supported in this animator.")
        numerator = power + 1
        body = f"x^({numerator})/({numerator})"
        if coeff == 1:
            return [body]
        if coeff == -1:
            return [f"-{body}"]
        return [f"{clean_number(coeff)}{body}"]
    if term.basis == "const":
        if coeff == 1:
            return ["x"]
        return [f"{clean_number(coeff)}x"]
    if term.basis == "sin":
        return ["-cos(x)" if coeff == 1 else f"{clean_number(-coeff)}cos(x)"]
    if term.basis == "cos":
        return ["sin(x)" if coeff == 1 else f"{clean_number(coeff)}sin(x)"]
    if term.basis == "exp":
        return ["exp(x)" if coeff == 1 else f"{clean_number(coeff)}exp(x)"]
    raise ValueError("Unsupported integral rule.")


def function_of_x_tokens(terms: list[SymbolicTerm], symbol: str) -> list[str]:
    tokens: list[str] = []
    for idx, term in enumerate(terms):
        if term.coeff == 0:
            continue
        if term.basis == "const":
            token = clean_number(term.coeff)
        elif term.basis == "pow":
            power = term.power or 1
            symbol_power = symbol if power == 1 else f"{symbol}^{power}"
            if term.coeff == 1:
                token = symbol_power
            elif term.coeff == -1:
                token = f"-{symbol_power}"
            elif abs(term.coeff).denominator != 1:
                abs_coeff = abs(term.coeff)
                if abs_coeff.numerator == 1:
                    token = f"{symbol_power}/{abs_coeff.denominator}"
                else:
                    token = f"{abs_coeff.numerator}{symbol_power}/{abs_coeff.denominator}"
                if term.coeff < 0:
                    token = f"-{token}"
            else:
                token = f"{clean_number(term.coeff)}{symbol_power}"
        else:
            token = clean_number(term.coeff) + term_body_text(term).replace("x", symbol) if term.coeff not in (1, -1) else (
                term_body_text(term).replace("x", symbol) if term.coeff == 1 else f"-{term_body_text(term).replace('x', symbol)}"
            )

        if idx == 0:
            tokens.append(token)
        else:
            if token.startswith("-"):
                tokens.append("-")
                tokens.append(token[1:])
            else:
                tokens.append("+")
                tokens.append(token)
    return tokens or ["0"]


def build_integral_steps(expression_text: str, lower: str | None = None, upper: str | None = None) -> list[SolveStep]:
    terms = parse_expression_terms(expression_text)
    int_tok = integral_token(lower, upper)
    steps = [SolveStep("Starting integral", [int_tok, "("] + symbolic_terms_to_tokens(terms) + [")", "dx"])]

    if len(terms) > 1:
        steps.append(SolveStep("Integrate term by term", op_tokens_for_terms(int_tok, terms, ["dx"])))

    raw_result: list[SymbolicTerm] = []
    solved_groups: list[list[str]] = []
    for idx, term in enumerate(terms):
        raw_result.extend(integral_of_term(term))
        raw_group = integral_rule_tokens(term)
        steps.append(
            SolveStep(
                f"Apply integral rule to {format_symbolic_term(abs_term(term))}",
                progressive_term_expression(terms, idx, solved_groups, raw_group, lambda pending: integral_piece_tokens(pending, int_tok)),
            )
        )
        simplified_group = symbolic_terms_to_tokens(integral_of_term(abs_term(term)))
        if simplified_group != raw_group:
            steps.append(
                SolveStep(
                    f"Simplify antiderivative of {format_symbolic_term(abs_term(term))}",
                    progressive_term_expression(terms, idx, solved_groups, simplified_group, lambda pending: integral_piece_tokens(pending, int_tok)),
                )
            )
        solved_groups.append(simplified_group)

    combined_raw = join_signed_groups(solved_groups, [-1 if term.coeff < 0 else 1 for term in terms])
    if combined_raw != steps[-1].tokens:
        steps.append(SolveStep("Collect antiderivative terms", combined_raw))

    simplified = canonicalize_terms(raw_result)
    simplified_tokens = symbolic_terms_to_tokens(simplified)
    if simplified_tokens != combined_raw:
        steps.append(SolveStep("Simplify", simplified_tokens))

    if lower is not None and upper is not None:
        lower_value = parse_constant_fraction(lower)
        upper_value = parse_constant_fraction(upper)
        steps.append(SolveStep("Name the antiderivative", ["F(x)", "="] + simplified_tokens))
        steps.append(SolveStep("Use the Fundamental Theorem", [f"F({clean_number(upper_value)})", "-", f"F({clean_number(lower_value)})"]))
        steps.append(SolveStep("Substitute the upper bound", [f"F({clean_number(upper_value)})", "="] + function_of_x_tokens(simplified, clean_number(upper_value))))
        steps.append(SolveStep("Simplify the upper bound", [f"F({clean_number(upper_value)})", "="] + value_tokens(simplified, upper_value)))
        steps.append(SolveStep("Substitute the lower bound", [f"F({clean_number(lower_value)})", "="] + function_of_x_tokens(simplified, clean_number(lower_value))))
        steps.append(SolveStep("Simplify the lower bound", [f"F({clean_number(lower_value)})", "="] + value_tokens(simplified, lower_value)))
        steps.append(SolveStep("Subtract lower from upper", value_tokens(simplified, upper_value) + ["-"] + value_tokens(simplified, lower_value)))
        result = sum((evaluate_term_at(term, upper_value) - evaluate_term_at(term, lower_value) for term in simplified), Fraction(0))
        steps.append(SolveStep("Definite integral", [clean_number(result)]))
        return steps

    steps.append(SolveStep("Add the constant of integration", simplified_tokens + ["+", "C"]))
    steps.append(SolveStep("Integral", simplified_tokens + ["+", "C"]))
    return steps


def build_limit_steps(variable_name: str, target_text: str, expression_text: str) -> list[SolveStep]:
    node = parse_numeric_expression(expression_text)
    target_lower = target_text.lower()
    if target_lower in {"inf", "+inf", "infinity", "+infinity"}:
        target_value = math.inf
        sample_points = [10.0, 50.0, 100.0, 500.0]
        direction = "positive"
    elif target_lower in {"-inf", "-infinity"}:
        target_value = -math.inf
        sample_points = [-10.0, -50.0, -100.0, -500.0]
        direction = "negative"
    else:
        target_value = float(parse_constant_fraction(target_text))
        direction = "finite"
        deltas = [1.0, 0.5, 0.1, 0.01]
        sample_points = deltas

    steps = [SolveStep("Starting limit", [f"lim[{variable_name}->{target_text}]", expression_text])]

    left_values: list[float] = []
    right_values: list[float] = []
    if direction == "finite":
        for delta in sample_points:
            left_x = target_value - delta
            right_x = target_value + delta
            left_val = evaluate_numeric_expression(node, variable_name, left_x)
            right_val = evaluate_numeric_expression(node, variable_name, right_x)
            left_values.append(left_val)
            right_values.append(right_val)
            steps.append(
                SolveStep(
                    f"Move {variable_name} closer by {clean_decimal(delta, 2)}",
                    [
                        f"{variable_name}={clean_decimal(left_x)}",
                        "=>",
                        clean_decimal(left_val),
                        "and",
                        f"{variable_name}={clean_decimal(right_x)}",
                        "=>",
                        clean_decimal(right_val),
                    ],
                )
            )

        last_left = left_values[-1]
        last_right = right_values[-1]
        if math.isfinite(last_left) and math.isfinite(last_right) and abs(last_left - last_right) < 1e-3:
            estimate = (last_left + last_right) / 2.0
            result_text = str(int(round(estimate))) if abs(estimate - round(estimate)) < 5e-3 else clean_decimal(estimate)
        elif abs(last_left) > 1e6 and abs(last_right) > 1e6 and math.copysign(1.0, last_left) == math.copysign(1.0, last_right):
            result_text = "inf" if last_left > 0 else "-inf"
        else:
            result_text = "DNE"

        steps.append(SolveStep("Compare both sides", [clean_decimal(last_left), "vs", clean_decimal(last_right)]))
        steps.append(SolveStep("Limit", [f"lim[{variable_name}->{target_text}]", "=", result_text]))
        return steps

    sampled_values: list[float] = []
    for point in sample_points:
        value = evaluate_numeric_expression(node, variable_name, point)
        sampled_values.append(value)
        steps.append(
            SolveStep(
                f"Push {variable_name} toward {target_text}",
                [f"{variable_name}={clean_decimal(point)}", "=>", clean_decimal(value)],
            )
        )
    last_value = sampled_values[-1]
    prev_value = sampled_values[-2]
    if math.isfinite(last_value) and abs(last_value - prev_value) < 1e-3:
        result_text = str(int(round(last_value))) if abs(last_value - round(last_value)) < 5e-3 else clean_decimal(last_value)
    elif abs(last_value) > 1e6 and math.copysign(1.0, last_value) == math.copysign(1.0, prev_value):
        result_text = "inf" if last_value > 0 else "-inf"
    else:
        result_text = clean_decimal(last_value)
    steps.append(SolveStep("Limit", [f"lim[{variable_name}->{target_text}]", "=", result_text]))
    return steps


def build_sum_steps(variable_name: str, start: int, end: int, expression_text: str) -> list[SolveStep]:
    node = parse_numeric_expression(expression_text)
    direction = 1 if end >= start else -1
    indices = list(range(start, end + direction, direction))
    if len(indices) > 16:
        raise ValueError("Sigma animation currently supports up to 16 terms.")

    steps = [SolveStep("Starting sum", [f"sum[{variable_name}={start},{end}]", expression_text])]
    values = [evaluate_numeric_expression(node, variable_name, idx) for idx in indices]
    steps.append(SolveStep("Evaluate each term", [clean_decimal(value) for value in values[0:1]]))

    expanded_tokens: list[str] = []
    for idx, value in enumerate(values):
        if idx:
            expanded_tokens.append("+")
        expanded_tokens.append(clean_decimal(value))
    steps.append(SolveStep("Expand the sigma sum", expanded_tokens))

    running_total = 0.0
    for idx, value in enumerate(values):
        previous = running_total
        running_total += value
        steps.append(
            SolveStep(
                f"Add term {idx + 1}",
                [clean_decimal(previous), "+", clean_decimal(value), "=", clean_decimal(running_total)],
            )
        )
    steps.append(SolveStep("Sigma sum", [clean_decimal(running_total)]))
    return steps


GRAPH_COLORS = [
    (90, 90, 255),
    (255, 210, 90),
    (110, 235, 160),
    (255, 150, 90),
    (220, 120, 255),
]


def build_graph_spec(expression_texts: list[str]) -> GraphSpec:
    nodes = [parse_numeric_expression(expression_text) for expression_text in expression_texts]
    x_values = np.linspace(-10.0, 10.0, 360)
    curves: list[GraphCurve] = []
    finite_values: list[float] = []
    for idx, (expression_text, node) in enumerate(zip(expression_texts, nodes)):
        y_values: list[float | None] = []
        for x in x_values:
            try:
                value = evaluate_numeric_expression(node, "x", float(x))
            except Exception:
                value = float("nan")
            if math.isfinite(value) and abs(value) < 1e6:
                y_values.append(value)
                finite_values.append(value)
            else:
                y_values.append(None)
        curves.append(GraphCurve(expression_text=expression_text, y_values=y_values, color=GRAPH_COLORS[idx % len(GRAPH_COLORS)]))
    if not finite_values:
        raise ValueError("Could not sample this function for graphing.")

    low = float(np.percentile(finite_values, 10))
    high = float(np.percentile(finite_values, 90))
    if abs(high - low) < 1e-6:
        center = finite_values[0]
        low = center - 1.0
        high = center + 1.0
    pad = max(1.0, (high - low) * 0.18)
    return GraphSpec(
        expressions_text=", ".join(expression_texts),
        x_values=[float(x) for x in x_values],
        curves=curves,
        x_min=-10.0,
        x_max=10.0,
        y_min=low - pad,
        y_max=high + pad,
    )


def build_graph_steps(expression_texts: list[str]) -> tuple[list[SolveStep], GraphSpec]:
    graph_spec = build_graph_spec(expression_texts)
    if len(expression_texts) == 1:
        lead = expression_texts[0]
        sample_tokens = ["y", "=", expression_texts[0]]
        plot_tokens = ["y", "=", expression_texts[0]]
    else:
        lead = ", ".join(expression_texts)
        sample_tokens = [str(len(expression_texts)), "curves"]
        plot_tokens = ["draw", str(len(expression_texts)), "curves"]
    steps = [
        SolveStep("Starting graph", ["graph", lead]),
        SolveStep("Set the graph window", ["x", "from", "-10", "to", "10"]),
        SolveStep("Sample the curve" if len(expression_texts) == 1 else "Sample curves", sample_tokens),
        SolveStep("Plot the curve" if len(expression_texts) == 1 else "Plot curves", plot_tokens),
        SolveStep("Graph", ["y", "="] + [lead]),
    ]
    return steps, graph_spec


def equation_tokens_with_operation(left: LinearExpr, right: LinearExpr, op_tokens: list[str]) -> list[str]:
    return format_side(left) + op_tokens + ["="] + format_side(right) + op_tokens


def build_steps(equation: str) -> list[SolveStep]:
    left, right = parse_equation(equation)
    steps = [SolveStep("Starting equation", format_equation(left, right))]

    if not is_zero(right.coef):
        moved_coef = right.coef
        op_tokens = ["-" if moved_coef > 0 else "+", format_term(abs(moved_coef))]
        steps.append(SolveStep(f"Apply {''.join(op_tokens)} to both sides", equation_tokens_with_operation(left, right, op_tokens)))
        left = LinearExpr(left.coef - right.coef, left.const)
        right = LinearExpr(Fraction(0), right.const)
        steps.append(SolveStep("Combine like x terms", format_equation(left, right)))

    if not is_zero(left.const):
        moved_const = left.const
        op_tokens = ["-" if moved_const > 0 else "+", clean_number(abs(moved_const))]
        steps.append(SolveStep(f"Apply {''.join(op_tokens)} to both sides", equation_tokens_with_operation(left, right, op_tokens)))
        right = LinearExpr(Fraction(0), right.const - left.const)
        left = LinearExpr(left.coef, Fraction(0))
        steps.append(SolveStep("Combine constants", format_equation(left, right)))

    if is_zero(left.coef):
        if is_zero(right.const):
            steps.append(SolveStep("Identity", ["all", "real", "x", "work"]))
        else:
            steps.append(SolveStep("Contradiction", ["no", "solution"]))
        return steps

    if not is_one(left.coef):
        divisor = clean_number(left.coef)
        op_tokens = ["/", divisor]
        steps.append(SolveStep(f"Divide both sides by {divisor}", equation_tokens_with_operation(left, right, op_tokens)))
        steps.append(SolveStep("Cancel the x coefficient", ["x", "="] + format_side(right) + ["/", divisor]))
        right = LinearExpr(Fraction(0), right.const / left.coef)
        left = LinearExpr(Fraction(1), Fraction(0))
        steps.append(SolveStep("Simplify both sides", format_equation(left, right)))

    steps.append(SolveStep("Solution", ["x", "=", clean_number(right.const)]))
    return steps


def parse_integral_request(text: str) -> tuple[str | None, str | None, str]:
    raw = text.strip()
    match = re.match(r"^(?:int|integral)\[(.+?),(.+?)\]\s+(.+)$", raw, re.IGNORECASE)
    if match:
        lower = match.group(1).strip()
        upper = match.group(2).strip()
        expr = match.group(3).strip()
        return lower, upper, expr
    raw_lower = raw.lower()
    if raw_lower.startswith("int "):
        return None, None, raw[4:].strip()
    if raw_lower.startswith("integral "):
        return None, None, raw[9:].strip()
    raise ValueError("Integral input must start with int or integral.")


def build_problem_steps(text: str) -> ProblemSpec:
    raw = text.strip()
    normalized = normalize_input_text(raw)
    lowered = normalized.lower()

    if lowered.startswith("diff "):
        expr = normalized[5:].strip()
        return ProblemSpec(build_derivative_steps(expr), f"diff {expr}", "d/dx x^n = n x^(n-1)")
    if lowered.startswith("derivative "):
        expr = normalized[11:].strip()
        return ProblemSpec(build_derivative_steps(expr), f"derivative {expr}", "d/dx x^n = n x^(n-1)")
    if lowered.startswith("d/dx "):
        expr = normalized[5:].strip()
        return ProblemSpec(build_derivative_steps(expr), f"d/dx {expr}", "d/dx x^n = n x^(n-1)")
    if lowered.startswith("lim[") or lowered.startswith("limit["):
        variable_name, target_text, expr_raw = parse_limit_request(raw)
        expr = normalize_input_text(expr_raw)
        shown = f"lim[{variable_name}->{target_text}] {expr}"
        return ProblemSpec(build_limit_steps(variable_name, target_text, expr), shown, "lim x->a f(x)")
    if lowered.startswith("sum[") or lowered.startswith("sigma["):
        variable_name, start, end, expr_raw = parse_sum_request(raw)
        expr = normalize_input_text(expr_raw)
        shown = f"sum[{variable_name}={start},{end}] {expr}"
        return ProblemSpec(build_sum_steps(variable_name, start, end, expr), shown, "sum[i=m,n] f(i)")
    if lowered.startswith("graph ") or lowered.startswith("plot "):
        expr_raw = parse_graph_request(raw)
        expressions = [normalize_input_text(strip_graph_prefix(piece)) for piece in split_top_level_commas(expr_raw)]
        if not expressions:
            raise ValueError("Provide at least one expression to graph.")
        steps, graph_spec = build_graph_steps(expressions)
        display_expr = ", ".join(expressions)
        formula = "graph y = f(x)" if len(expressions) == 1 else "graph y = f(x), g(x), h(x)"
        return ProblemSpec(steps, f"graph {display_expr}", formula, graph_spec=graph_spec)
    raw_lower = raw.lower()
    if raw_lower.startswith("int") or raw_lower.startswith("integral"):
        lower, upper, expr_raw = parse_integral_request(raw)
        expr = normalize_input_text(expr_raw)
        shown = f"int[{lower},{upper}] {expr}" if lower is not None and upper is not None else f"int {expr}"
        formula = "int[a,b] f(x) dx = F(b) - F(a)" if lower is not None and upper is not None else "int x^n dx = x^(n+1)/(n+1) + C"
        return ProblemSpec(build_integral_steps(expr, lower, upper), shown, formula)
    if "=" in normalized:
        quadratic_steps = build_quadratic_steps(normalized)
        if quadratic_steps is not None:
            return ProblemSpec(quadratic_steps, normalized, "ax^2 + bx + c = 0")
        return ProblemSpec(build_steps(normalized), normalized, "ax + b = c  ->  x = (c - b)/a")
    raise ValueError("Use an equation, or start with diff, int, lim, sum, or graph.")


def token_size(token: str) -> tuple[int, int]:
    sprite = get_text_sprite(token, TEXT, EQUATION_SCALE)
    return sprite.width, sprite.height


def layout_tokens(tokens: list[str], y: int | None = None) -> list[tuple[int, int]]:
    widths = [token_size(token)[0] for token in tokens]
    total_width = sum(widths) + TOKEN_GAP * max(0, len(tokens) - 1)
    x = (WIDTH - total_width) // 2
    baseline_y = HEIGHT // 2 + 8 if y is None else y
    positions: list[tuple[int, int]] = []
    for width in widths:
        positions.append((x, baseline_y))
        x += width + TOKEN_GAP
    return positions


def layout_tokens_scaled(tokens: list[str], y: int, scale: float, gap: int) -> list[tuple[int, int]]:
    widths = [get_text_sprite(token, TEXT, scale).width for token in tokens]
    total_width = sum(widths) + gap * max(0, len(tokens) - 1)
    x = (WIDTH - total_width) // 2
    positions: list[tuple[int, int]] = []
    for width in widths:
        positions.append((x, y))
        x += width + gap
    return positions


def layout_tokens_left(tokens: list[str], x: int, y: int, scale: float, gap: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    cursor_x = x
    for token in tokens:
        sprite = get_text_sprite(token, TEXT, scale)
        positions.append((cursor_x, y))
        cursor_x += sprite.width + gap
    return positions


def match_tokens(old_tokens: list[str], new_tokens: list[str]) -> tuple[dict[int, int], set[int], set[int]]:
    matches: dict[int, int] = {}
    used_new: set[int] = set()
    for i, old in enumerate(old_tokens):
        for j, new in enumerate(new_tokens):
            if j in used_new:
                continue
            if old == new:
                matches[i] = j
                used_new.add(j)
                break
    old_only = {i for i in range(len(old_tokens)) if i not in matches}
    new_only = {j for j in range(len(new_tokens)) if j not in used_new}
    return matches, old_only, new_only


def get_text_sprite(text: str, color: tuple[int, int, int], scale: float) -> TextSprite:
    key = (text, color, scale)
    cached = TEXT_CACHE.get(key)
    if cached is not None:
        return cached

    if text.startswith("INT|"):
        _, lower, upper = text.split("|", 2)
        sprite = create_integral_sprite(lower or None, upper or None, color, scale)
        TEXT_CACHE[key] = sprite
        return sprite

    if "^" in text:
        sprite = create_superscript_sprite(text, color, scale)
        TEXT_CACHE[key] = sprite
        return sprite

    (w, h), baseline = cv2.getTextSize(text, FONT, scale, THICKNESS)
    width = max(1, w + 4)
    height = max(1, h + baseline + 4)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.putText(mask, text, (2, h + 2), FONT, scale, 255, THICKNESS, cv2.LINE_AA)

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = color
    sprite = TextSprite(image=image, alpha=mask.astype(np.float32) / 255.0, baseline=h + 2, width=width, height=height)
    TEXT_CACHE[key] = sprite
    return sprite


def create_plain_text_sprite(text: str, color: tuple[int, int, int], scale: float) -> TextSprite:
    (w, h), baseline = cv2.getTextSize(text, FONT, scale, THICKNESS)
    width = max(1, w + 4)
    height = max(1, h + baseline + 4)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.putText(mask, text, (2, h + 2), FONT, scale, 255, THICKNESS, cv2.LINE_AA)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = color
    return TextSprite(image=image, alpha=mask.astype(np.float32) / 255.0, baseline=h + 2, width=width, height=height)


def split_superscript_token(text: str) -> tuple[str, str, str]:
    caret = text.find("^")
    if caret == -1:
        return text, "", ""
    base = text[:caret]
    rest = text[caret + 1 :]
    if rest.startswith("("):
        depth = 0
        for idx, ch in enumerate(rest):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    exponent = rest[1:idx]
                    tail = rest[idx + 1 :]
                    return base, exponent, tail
        return base, rest, ""

    match = re.match(r"([A-Za-z0-9+\-]+)(.*)$", rest)
    if match:
        return base, match.group(1), match.group(2)
    return base, rest, ""


def blit_sprite(dest_image: np.ndarray, sprite: TextSprite, x0: int, y0: int, alpha: float = 1.0) -> None:
    x1 = x0 + sprite.width
    y1 = y0 + sprite.height

    clip_x0 = max(0, x0)
    clip_y0 = max(0, y0)
    clip_x1 = min(dest_image.shape[1], x1)
    clip_y1 = min(dest_image.shape[0], y1)
    if clip_x0 >= clip_x1 or clip_y0 >= clip_y1:
        return

    src_x0 = clip_x0 - x0
    src_y0 = clip_y0 - y0
    src_x1 = src_x0 + (clip_x1 - clip_x0)
    src_y1 = src_y0 + (clip_y1 - clip_y0)

    src_img = sprite.image[src_y0:src_y1, src_x0:src_x1].astype(np.float32)
    src_alpha = sprite.alpha[src_y0:src_y1, src_x0:src_x1] * float(alpha)
    roi = dest_image[clip_y0:clip_y1, clip_x0:clip_x1].astype(np.float32)
    src_alpha_3 = src_alpha[..., None]
    blended = roi * (1.0 - src_alpha_3) + src_img * src_alpha_3
    dest_image[clip_y0:clip_y1, clip_x0:clip_x1] = blended.astype(np.uint8)


def create_superscript_sprite(text: str, color: tuple[int, int, int], scale: float) -> TextSprite:
    base_text, exponent_text, tail_text = split_superscript_token(text)
    if not exponent_text:
        return create_plain_text_sprite(text, color, scale)

    base_sprite = create_plain_text_sprite(base_text, color, scale)
    exp_scale = scale * 0.62
    exp_sprite = create_plain_text_sprite(exponent_text, color, exp_scale)
    tail_sprite = create_plain_text_sprite(tail_text, color, scale) if tail_text else None

    exp_offset_y = 0
    base_offset_y = exp_sprite.height - max(4, int(scale * 6))
    tail_offset_y = base_offset_y
    exp_x = max(0, base_sprite.width - 4)
    tail_x = exp_x + exp_sprite.width + 2

    width = tail_x + (tail_sprite.width if tail_sprite else 0)
    height = max(base_offset_y + base_sprite.height, exp_offset_y + exp_sprite.height, tail_offset_y + (tail_sprite.height if tail_sprite else 0))
    image = np.zeros((height, width, 3), dtype=np.uint8)

    blit_sprite(image, base_sprite, 0, base_offset_y, 1.0)
    blit_sprite(image, exp_sprite, exp_x, exp_offset_y, 1.0)
    if tail_sprite:
        blit_sprite(image, tail_sprite, tail_x, tail_offset_y, 1.0)

    alpha = np.max(image.astype(np.float32) / np.maximum(image.max(), 1), axis=2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    alpha = np.maximum(alpha, gray.astype(np.float32) / 255.0)
    baseline = base_offset_y + base_sprite.baseline
    return TextSprite(image=image, alpha=alpha, baseline=baseline, width=width, height=height)


def create_fallback_integral_mask(width: int, height: int, thickness: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    upper = cv2.ellipse2Poly(
        (int(width * 0.66), int(height * 0.15)),
        (int(width * 0.13), int(height * 0.08)),
        0,
        210,
        25,
        6,
    )
    lower = cv2.ellipse2Poly(
        (int(width * 0.34), int(height * 0.84)),
        (int(width * 0.13), int(height * 0.08)),
        0,
        30,
        205,
        6,
    )
    spine = np.array(
        [
            (int(width * 0.58), int(height * 0.21)),
            (int(width * 0.54), int(height * 0.32)),
            (int(width * 0.48), int(height * 0.55)),
            (int(width * 0.42), int(height * 0.78)),
        ],
        dtype=np.int32,
    )
    stroke = np.vstack([upper, spine, lower])
    cv2.polylines(mask, [stroke], False, 255, thickness, cv2.LINE_AA)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, thickness // 2), max(3, thickness // 2)))
    return cv2.dilate(mask, kernel, iterations=1)


def create_pillow_glyph_sprite(
    text: str,
    color: tuple[int, int, int],
    pixel_size: int,
    font_candidates: list[str],
    padding_x: int = 12,
    padding_y: int = 8,
) -> TextSprite | None:
    font_path = resolve_integral_font_path() if font_candidates == INTEGRAL_FONT_CANDIDATES else None
    if font_path is None and font_candidates != INTEGRAL_FONT_CANDIDATES:
        for path in font_candidates:
            if os.path.exists(path):
                font_path = path
                break
    if not font_path:
        return None

    font = ImageFont.truetype(font_path, pixel_size)
    probe = Image.new("L", (1, 1), 0)
    probe_draw = ImageDraw.Draw(probe)
    bbox = probe_draw.textbbox((0, 0), text, font=font)
    width = max(1, bbox[2] - bbox[0] + padding_x * 2)
    height = max(1, bbox[3] - bbox[1] + padding_y * 2)

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.text((padding_x - bbox[0], padding_y - bbox[1]), text, fill=255, font=font)

    alpha = np.array(mask_img, dtype=np.float32) / 255.0
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = color
    baseline = min(height - 1, int(height * 0.82))
    return TextSprite(image=image, alpha=alpha, baseline=baseline, width=width, height=height)


def rescale_sprite(sprite: TextSprite, factor: float) -> TextSprite:
    if factor <= 0:
        return sprite
    new_width = max(1, int(round(sprite.width * factor)))
    new_height = max(1, int(round(sprite.height * factor)))
    if new_width == sprite.width and new_height == sprite.height:
        return sprite
    image = cv2.resize(sprite.image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(sprite.alpha, (new_width, new_height), interpolation=cv2.INTER_AREA)
    baseline = max(1, int(round(sprite.baseline * factor)))
    return TextSprite(image=image, alpha=alpha, baseline=baseline, width=new_width, height=new_height)


def create_integral_sprite(lower: str | None, upper: str | None, color: tuple[int, int, int], scale: float) -> TextSprite:
    bound_scale = scale * 0.24
    upper_sprite = create_plain_text_sprite(upper, color, bound_scale) if upper else None
    lower_sprite = create_plain_text_sprite(lower, color, bound_scale) if lower else None

    symbol_sprite = create_pillow_glyph_sprite("∫", color, max(52, int(scale * 46)), INTEGRAL_FONT_CANDIDATES, padding_x=4, padding_y=0)
    if symbol_sprite is None:
        symbol_w = max(22, int(scale * 12))
        symbol_h = max(62, int(scale * 34))
        thickness = max(2, int(scale * 1.4))
        mask = create_fallback_integral_mask(symbol_w, symbol_h, thickness)
        image = np.zeros((symbol_h, symbol_w, 3), dtype=np.uint8)
        image[:] = color
        symbol_sprite = TextSprite(image=image, alpha=mask.astype(np.float32) / 255.0, baseline=int(symbol_h * 0.82), width=symbol_w, height=symbol_h)

    text_probe = create_plain_text_sprite("x", color, scale)
    target_symbol_height = int(text_probe.height * (1.18 if not (upper or lower) else 1.32))
    if symbol_sprite.height > target_symbol_height:
        symbol_sprite = rescale_sprite(symbol_sprite, target_symbol_height / symbol_sprite.height)

    symbol_w = symbol_sprite.width
    symbol_h = symbol_sprite.height
    upper_h = upper_sprite.height if upper_sprite else 0
    lower_h = lower_sprite.height if lower_sprite else 0
    width = max(symbol_w, upper_sprite.width if upper_sprite else 0, lower_sprite.width if lower_sprite else 0) + 8
    height = upper_h + symbol_h + lower_h + 4
    image = np.zeros((height, width, 3), dtype=np.uint8)

    y_symbol = upper_h + 1
    symbol_x = (width - symbol_w) // 2
    blit_sprite(image, symbol_sprite, symbol_x, y_symbol, 1.0)

    if upper_sprite:
        upper_x = min(width - upper_sprite.width, max(0, symbol_x + symbol_w // 2 - upper_sprite.width // 2 + 5))
        blit_sprite(image, upper_sprite, upper_x, 0, 1.0)
    if lower_sprite:
        lower_x = min(width - lower_sprite.width, max(0, symbol_x + symbol_w // 2 - lower_sprite.width // 2 - 6))
        blit_sprite(image, lower_sprite, lower_x, upper_h + symbol_h + 2, 1.0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    alpha = gray.astype(np.float32) / 255.0
    baseline = y_symbol + int(symbol_h * 0.82)
    return TextSprite(image=image, alpha=alpha, baseline=baseline, width=width, height=height)


def draw_text_alpha(image: np.ndarray, text: str, position: tuple[int, int], color: tuple[int, int, int], alpha: float, scale: float = EQUATION_SCALE) -> None:
    if alpha <= 0.0:
        return
    sprite = get_text_sprite(text, color, scale)
    x0 = position[0]
    y0 = position[1] - sprite.baseline
    blit_sprite(image, sprite, x0, y0, alpha)


def ease_in_out(t: float) -> float:
    t = np.clip(t, 0.0, 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def stage_phase(frame_index: int, hold_frames: int, frames_per_step: int) -> float:
    total = hold_frames * 2 + frames_per_step
    if frame_index < hold_frames:
        return 0.0
    if frame_index >= hold_frames + frames_per_step:
        return 1.0
    raw = (frame_index - hold_frames) / max(1, frames_per_step - 1)
    return ease_in_out(raw)


def draw_chrome(image: np.ndarray, label: str, equation_text: str, formula_text: str, paused: bool, frames_per_step: int) -> None:
    cv2.rectangle(image, (72, 46), (WIDTH - 72, 128), PANEL, -1)
    cv2.rectangle(image, (72, 46), (WIDTH - 72, 128), PANEL_EDGE, 1)
    cv2.putText(image, "Equation Solver Animation", (WIDTH // 2 - 220, 84), FONT, TITLE_SCALE, TEXT, 1, cv2.LINE_AA)
    cv2.line(image, (WIDTH // 2 - 160, 96), (WIDTH // 2 + 160, 96), GRID, 1, cv2.LINE_AA)
    cv2.putText(image, formula_text, (WIDTH // 2 - min(360, 6 * len(formula_text)), 116), FONT, 0.56, MUTED, 1, cv2.LINE_AA)
    cv2.putText(image, label, (120, 164), FONT, 0.78, STEP_COLOR, 1, cv2.LINE_AA)
    cv2.rectangle(image, (72, HEIGHT - 122), (WIDTH - 72, HEIGHT - 38), PANEL, -1)
    cv2.rectangle(image, (72, HEIGHT - 122), (WIDTH - 72, HEIGHT - 38), PANEL_EDGE, 1)
    cv2.putText(image, equation_text, (104, HEIGHT - 88), FONT, 0.64, MUTED, 1, cv2.LINE_AA)
    speed_pct = int(round(DEFAULT_FRAMES_PER_STEP / max(1, frames_per_step) * 100))
    cv2.putText(image, f"speed {speed_pct}%   [ slower ] faster", (104, HEIGHT - 58), FONT, INFO_SCALE, MUTED, 1, cv2.LINE_AA)
    cv2.putText(image, "q quit   p pause   r restart   n new", (WIDTH - 340, HEIGHT - 58), FONT, INFO_SCALE, MUTED, 1, cv2.LINE_AA)
    cv2.putText(image, "PAUSED" if paused else "RUNNING", (WIDTH - 176, 84), FONT, INFO_SCALE, STEP_COLOR if paused else ACCENT, 1, cv2.LINE_AA)


def draw_step_history(image: np.ndarray, steps: list[SolveStep], upto_index: int, graph_mode: bool) -> None:
    visible_count = 2 if graph_mode else 4
    history = steps[max(0, upto_index - visible_count) : upto_index]
    if not history:
        return
    panel_x0 = 72
    panel_y0 = 242 if graph_mode else 188
    panel_x1 = 510 if graph_mode else 610
    panel_y1 = 474 if graph_mode else 500
    cv2.rectangle(image, (panel_x0, panel_y0), (panel_x1, panel_y1), PANEL, -1)
    cv2.rectangle(image, (panel_x0, panel_y0), (panel_x1, panel_y1), PANEL_EDGE, 1)
    cv2.putText(image, "Recent Steps", (panel_x0 + 20, panel_y0 + 28), FONT, 0.56, MUTED, 1, cv2.LINE_AA)
    base_y = panel_y0 + 56
    row_gap = 58 if graph_mode else 66
    for idx, step in enumerate(history):
        age = len(history) - 1 - idx
        alpha = max(0.18, 0.82 - age * 0.18)
        token_scale = max(0.50, 0.72 - age * 0.07)
        label_scale = max(0.40, 0.56 - age * 0.06)
        y = base_y + idx * row_gap
        draw_text_alpha(image, f"{step.label}:", (panel_x0 + 20, y), STEP_COLOR, alpha, scale=label_scale)
        token_y = y + 28
        positions = layout_tokens_left(step.tokens, panel_x0 + 20, token_y, token_scale, max(9, int(TOKEN_GAP * 0.5)))
        for token, pos in zip(step.tokens, positions):
            draw_text_alpha(image, token, pos, SOFT, alpha, scale=token_scale)


def draw_graph_panel(image: np.ndarray, graph_spec: GraphSpec, progress: float) -> None:
    x0, y0, x1, y1 = 560, 242, WIDTH - 72, HEIGHT - 142
    cv2.rectangle(image, (x0, y0), (x1, y1), PANEL, -1)
    cv2.rectangle(image, (x0, y0), (x1, y1), PANEL_EDGE, 1)
    cv2.putText(image, "Shared Graph", (x0 + 20, y0 + 28), FONT, 0.60, TEXT, 1, cv2.LINE_AA)

    legend_x = x0 + 160
    legend_y = y0 + 24
    for idx, curve in enumerate(graph_spec.curves[:4]):
        lx = legend_x + idx * 125
        cv2.circle(image, (lx, legend_y), 5, curve.color, -1, cv2.LINE_AA)
        cv2.putText(image, curve.expression_text, (lx + 14, legend_y + 5), FONT, 0.44, MUTED, 1, cv2.LINE_AA)

    plot_x0, plot_y0 = x0 + 26, y0 + 50
    plot_x1, plot_y1 = x1 - 24, y1 - 60
    cv2.rectangle(image, (plot_x0, plot_y0), (plot_x1, plot_y1), (24, 24, 24), -1)
    cv2.rectangle(image, (plot_x0, plot_y0), (plot_x1, plot_y1), (46, 46, 46), 1)

    def map_x(value: float) -> int:
        return int(plot_x0 + (value - graph_spec.x_min) / (graph_spec.x_max - graph_spec.x_min) * (plot_x1 - plot_x0))

    def map_y(value: float) -> int:
        return int(plot_y1 - (value - graph_spec.y_min) / (graph_spec.y_max - graph_spec.y_min) * (plot_y1 - plot_y0))

    if graph_spec.x_min <= 0.0 <= graph_spec.x_max:
        axis_x = map_x(0.0)
        cv2.line(image, (axis_x, plot_y0), (axis_x, plot_y1), GRID, 1, cv2.LINE_AA)
    if graph_spec.y_min <= 0.0 <= graph_spec.y_max:
        axis_y = map_y(0.0)
        cv2.line(image, (plot_x0, axis_y), (plot_x1, axis_y), GRID, 1, cv2.LINE_AA)

    for fraction in (0.25, 0.5, 0.75):
        gx = int(plot_x0 + (plot_x1 - plot_x0) * fraction)
        gy = int(plot_y0 + (plot_y1 - plot_y0) * fraction)
        cv2.line(image, (gx, plot_y0), (gx, plot_y1), (34, 34, 34), 1, cv2.LINE_AA)
        cv2.line(image, (plot_x0, gy), (plot_x1, gy), (34, 34, 34), 1, cv2.LINE_AA)

    visible_samples = max(2, int(len(graph_spec.x_values) * np.clip(progress, 0.0, 1.0)))
    for curve in graph_spec.curves:
        last_point: tuple[int, int] | None = None
        for x_value, y_value in zip(graph_spec.x_values[:visible_samples], curve.y_values[:visible_samples]):
            if y_value is None or not math.isfinite(y_value):
                last_point = None
                continue
            clipped = min(max(y_value, graph_spec.y_min), graph_spec.y_max)
            point = (map_x(x_value), map_y(clipped))
            if last_point is not None and abs(point[1] - last_point[1]) < (plot_y1 - plot_y0):
                cv2.line(image, last_point, point, curve.color, 2, cv2.LINE_AA)
            last_point = point

    cv2.putText(image, clean_decimal(graph_spec.x_min, 2), (plot_x0 - 8, plot_y1 + 38), FONT, 0.42, SOFT, 1, cv2.LINE_AA)
    cv2.putText(image, clean_decimal(graph_spec.x_max, 2), (plot_x1 - 22, plot_y1 + 38), FONT, 0.42, SOFT, 1, cv2.LINE_AA)


def draw_transition(
    all_steps: list[SolveStep],
    step_index: int,
    old_step: SolveStep,
    new_step: SolveStep,
    equation_text: str,
    formula_text: str,
    frame_index: int,
    paused: bool,
    hold_frames: int,
    frames_per_step: int,
    graph_spec: GraphSpec | None = None,
) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), BG, dtype=np.uint8)
    draw_chrome(image, new_step.label, equation_text, formula_text, paused, frames_per_step)
    draw_step_history(image, all_steps, step_index, graph_spec is not None)
    phase = stage_phase(frame_index, hold_frames, frames_per_step)
    if graph_spec is not None:
        overall_progress = (step_index + phase) / max(1, len(all_steps) - 1)
        draw_graph_panel(image, graph_spec, overall_progress)

    equation_y = 204 if graph_spec is not None else 588
    if graph_spec is not None:
        token_scale = 1.08
        gap = 12
        old_pos = layout_tokens_scaled(old_step.tokens, equation_y, token_scale, gap)
        new_pos = layout_tokens_scaled(new_step.tokens, equation_y, token_scale, gap)
    else:
        token_scale = EQUATION_SCALE
        gap = TOKEN_GAP
        old_pos = layout_tokens(old_step.tokens, y=equation_y)
        new_pos = layout_tokens(new_step.tokens, y=equation_y)
    matches, old_only, new_only = match_tokens(old_step.tokens, new_step.tokens)

    for old_i, new_i in matches.items():
        start = old_pos[old_i]
        end = new_pos[new_i]
        x = int(start[0] + (end[0] - start[0]) * phase)
        y = int(start[1] + (end[1] - start[1]) * phase)
        draw_text_alpha(image, new_step.tokens[new_i], (x, y), TEXT, 1.0, scale=token_scale)

    for old_i in old_only:
        start = old_pos[old_i]
        y = int(start[1] - 40 * phase)
        fade = 1.0 - phase
        draw_text_alpha(image, old_step.tokens[old_i], (start[0], y), MUTED, fade, scale=token_scale)

    for new_i in new_only:
        end = new_pos[new_i]
        y = int(end[1] + 40 * (1.0 - phase))
        draw_text_alpha(image, new_step.tokens[new_i], (end[0], y), ACCENT, phase, scale=token_scale)

    return image


def draw_static_step(
    all_steps: list[SolveStep],
    step_index: int,
    step: SolveStep,
    equation_text: str,
    formula_text: str,
    paused: bool,
    frames_per_step: int,
    graph_spec: GraphSpec | None = None,
) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), BG, dtype=np.uint8)
    draw_chrome(image, step.label, equation_text, formula_text, paused, frames_per_step)
    draw_step_history(image, all_steps, step_index, graph_spec is not None)
    if graph_spec is not None:
        progress = step_index / max(1, len(all_steps) - 1)
        draw_graph_panel(image, graph_spec, progress)
    equation_y = 204 if graph_spec is not None else 588
    if graph_spec is not None:
        token_scale = 1.08
        positions = layout_tokens_scaled(step.tokens, equation_y, token_scale, 12)
    else:
        token_scale = EQUATION_SCALE
        positions = layout_tokens(step.tokens, y=equation_y)
    for token, pos in zip(step.tokens, positions):
        draw_text_alpha(image, token, pos, TEXT, 1.0, scale=token_scale)
    return image


def draw_editor(equation_text: str, cursor_index: int, message: str | None = None) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), BG, dtype=np.uint8)
    cv2.rectangle(image, (72, 46), (WIDTH - 72, 136), PANEL, -1)
    cv2.rectangle(image, (72, 46), (WIDTH - 72, 136), PANEL_EDGE, 1)
    cv2.putText(image, "Equation Solver Animation", (WIDTH // 2 - 220, 92), FONT, TITLE_SCALE, TEXT, 1, cv2.LINE_AA)
    cv2.line(image, (WIDTH // 2 - 180, 106), (WIDTH // 2 + 180, 106), GRID, 1, cv2.LINE_AA)
    cv2.putText(image, "Type equations, calculus, limits, sums, and graphs in one place", (WIDTH // 2 - 430, 170), FONT, 0.66, STEP_COLOR, 1, cv2.LINE_AA)
    cv2.rectangle(image, (150, HEIGHT // 2 - 74), (WIDTH - 150, HEIGHT // 2 + 42), PANEL, -1)
    cv2.rectangle(image, (150, HEIGHT // 2 - 74), (WIDTH - 150, HEIGHT // 2 + 42), INPUT_BORDER, 1)
    cv2.putText(image, equation_text or " ", (182, HEIGHT // 2 + 4), FONT, 1.1, TEXT, 2, cv2.LINE_AA)

    cursor_prefix = equation_text[:cursor_index]
    cursor_x = 182 + cv2.getTextSize(cursor_prefix, FONT, 1.1, 2)[0][0] + 6
    cv2.line(image, (cursor_x, HEIGHT // 2 - 42), (cursor_x, HEIGHT // 2 + 12), ACCENT, 2, cv2.LINE_AA)

    cv2.rectangle(image, (150, HEIGHT // 2 + 88), (WIDTH - 150, HEIGHT // 2 + 206), PANEL, -1)
    cv2.rectangle(image, (150, HEIGHT // 2 + 88), (WIDTH - 150, HEIGHT // 2 + 206), PANEL_EDGE, 1)
    cv2.putText(image, "Examples", (176, HEIGHT // 2 + 118), FONT, 0.58, MUTED, 1, cv2.LINE_AA)
    cv2.putText(image, "solve 2x+3=11     x^2+5x+6=0     diff x^3+2x     int[0,1] x^2", (176, HEIGHT // 2 + 152), FONT, 0.50, SOFT, 1, cv2.LINE_AA)
    cv2.putText(image, "lim[x->0] sin(x)/x     sum[i=1,4] i^2     graph y=x^2, x+1, sin(x)", (176, HEIGHT // 2 + 184), FONT, 0.54, SOFT, 1, cv2.LINE_AA)
    cv2.putText(image, "Enter solve   arrows move cursor   Backspace delete   q or Esc quit", (WIDTH // 2 - 310, HEIGHT - 52), FONT, INFO_SCALE, MUTED, 1, cv2.LINE_AA)
    cv2.putText(image, "Press n for a fresh blank editor", (WIDTH // 2 - 145, HEIGHT - 80), FONT, INFO_SCALE, MUTED, 1, cv2.LINE_AA)
    if message:
        cv2.putText(image, message, (WIDTH // 2 - min(420, 7 * len(message)), HEIGHT // 2 + 252), FONT, 0.64, ERROR, 1, cv2.LINE_AA)
    return image


def normalize_key(key: int) -> str:
    if key == 27:
        return "ESC"
    if key in (10, 13):
        return "ENTER"
    if key in (8, 127):
        return "BACKSPACE"
    if key in (2424832, 81, 65361, 63234):
        return "LEFT"
    if key in (2555904, 83, 65363, 63235):
        return "RIGHT"
    if key in (2490368, 82, 65362, 63232):
        return "UP"
    if key in (2621440, 84, 65364, 63233):
        return "DOWN"
    if key in (3014656, 65535, 63272):
        return "DELETE"
    if key in (63273,):
        return "HOME"
    if key in (63275,):
        return "END"
    low = key & 0xFF
    if 32 <= low <= 126:
        ch = chr(low)
        return ch.lower() if ch.isalpha() else ch
    return ""


def main() -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

    equation_text = DEFAULT_EQUATION
    error_message: str | None = None
    mode = "edit"
    steps: list[SolveStep] = []
    step_index = 0
    frame_index = 0
    paused = False
    frames_per_step = DEFAULT_FRAMES_PER_STEP
    hold_frames = DEFAULT_HOLD_FRAMES
    cursor_index = len(equation_text)
    formula_text = ""
    graph_spec: GraphSpec | None = None

    while True:
        if mode == "edit":
            frame = draw_editor(equation_text, cursor_index, error_message)
        elif step_index >= len(steps) - 1:
            frame = draw_static_step(steps, max(0, len(steps) - 1), steps[-1], equation_text, formula_text, paused, frames_per_step, graph_spec)
        else:
            frame = draw_transition(steps, step_index, steps[step_index], steps[step_index + 1], equation_text, formula_text, frame_index, paused, hold_frames, frames_per_step, graph_spec)

        cv2.imshow(WINDOW_NAME, frame)
        key = normalize_key(cv2.waitKeyEx(FRAME_DELAY_MS))

        if key in ("q", "ESC"):
            cv2.destroyAllWindows()
            return

        if mode == "edit":
            if key == "":
                continue
            if key == "ENTER":
                try:
                    problem = build_problem_steps(equation_text)
                    steps = problem.steps
                    equation_text = problem.display_text
                    formula_text = problem.formula_text
                    graph_spec = problem.graph_spec
                    error_message = None
                    mode = "play"
                    paused = False
                    step_index = 0
                    frame_index = 0
                except Exception as exc:
                    error_message = str(exc)
            elif key == "BACKSPACE":
                if cursor_index > 0:
                    equation_text = equation_text[: cursor_index - 1] + equation_text[cursor_index:]
                    cursor_index -= 1
            elif key == "DELETE":
                if cursor_index < len(equation_text):
                    equation_text = equation_text[:cursor_index] + equation_text[cursor_index + 1 :]
            elif key == "LEFT":
                cursor_index = max(0, cursor_index - 1)
            elif key == "RIGHT":
                cursor_index = min(len(equation_text), cursor_index + 1)
            elif key == "UP":
                cursor_index = 0
            elif key == "DOWN":
                cursor_index = len(equation_text)
            elif key == "HOME":
                cursor_index = 0
            elif key == "END":
                cursor_index = len(equation_text)
            elif len(key) == 1 and 32 <= ord(key) <= 126:
                equation_text = equation_text[:cursor_index] + key + equation_text[cursor_index:]
                cursor_index += 1
            continue

        if key in ("p", " "):
            paused = not paused
        elif key == "r":
            step_index = 0
            frame_index = 0
            paused = False
        elif key == "n":
            mode = "edit"
            error_message = None
            paused = False
            equation_text = ""
            cursor_index = 0
            graph_spec = None
            continue
        elif key == "[":
            frames_per_step = min(MAX_FRAMES_PER_STEP, frames_per_step + SPEED_STEP)
            hold_frames = max(8, frames_per_step // 7)
        elif key == "]":
            frames_per_step = max(MIN_FRAMES_PER_STEP, frames_per_step - SPEED_STEP)
            hold_frames = max(8, frames_per_step // 7)

        if not paused and step_index < len(steps) - 1:
            frame_index += 1
            if frame_index >= hold_frames * 2 + frames_per_step:
                frame_index = 0
                step_index += 1


if __name__ == "__main__":
    main()
