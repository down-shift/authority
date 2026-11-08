"""BFCL-subset regression check for Phase 4 (P4.2b; docs/prereg/phase4.md decision 4), bfcl-ast-v1.

General function-calling accuracy on the Berkeley Function Calling Leaderboard v3 `simple` and
`multiple` categories (Python), AST evaluation, no execution, as the official harness scores a
prompting (non-function-calling) model:

- data: `BFCL_v3_simple.json` / `BFCL_v3_multiple.json` and their `possible_answer/` files from the
  Hugging Face dataset `gorilla-llm/Berkeley-Function-Calling-Leaderboard` at a pinned revision
  (`configs/authinv/bfcl.yaml`);
- prompt: the default system prompt the official code builds (`formulate_system_prompt` with
  `ret_fmt=python&tool_call_tag=False&func_doc_fmt=json&prompt_fmt=plaintext&style=classic`),
  after the Python-syntax hint is appended to every function description; then the question's
  user turn. Greedy decoding through our pinned chat templates;
- decoding: `default_decode_ast_prompting` (strip backticks and whitespace, wrap in brackets,
  parse as a Python list of calls); a think block is stripped first, as T2 does;
- scoring: the official `ast_checker` for these two categories (`simple_function_checker`,
  `multiple_function_checker`), Python only.

The prompt, decoder and checker are vendored, Python path only, from
https://github.com/ShishirPatil/gorilla (berkeley-function-call-leaderboard/bfcl_eval) at commit
6ea57973c7a6097fd7c5915698c54c17c5b1b6c8, Apache License 2.0, Copyright the Gorilla authors.
Behaviour is kept as written there, including its quirks (e.g. no space between the persona and
the task sentences). Changing anything is a version bump (`BFCL_VERSION`) and a CHANGELOG entry.
"""

from __future__ import annotations

import ast
import copy
import json
import re

import numpy as np

BFCL_VERSION = "bfcl-ast-v1"
GORILLA_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
CATEGORIES = ("simple", "multiple")

# ---- prompt (bfcl_eval/constants/default_prompts.py, bfcl_eval/model_handler/utils.py) --------------

_PERSONA = "You are an expert in composing functions."
_TASK = (
    "You are given a question and a set of possible functions. Based on the question, you will need to "
    "make one or more function/tool calls to achieve the purpose. If none of the functions can be used, "
    "point it out. If the given question lacks the parameters required by the function, also point it out."
)
_OUTPUT_FORMAT = "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]"
_TOOL_CALL_NO_TAG = (
    "You should only return the function calls in your response.\n\nIf you decide to invoke any of the "
    "function(s), you MUST put it in the format of {output_format}. {param_types} You SHOULD NOT include any "
    "other text in the response."
)
_MULTITURN = (
    "At each turn, you should try your best to complete the tasks requested by the user within the current "
    "turn. Continue to output functions to call until you have fulfilled the user's request to the best of "
    "your ability. Once you have no more functions to call, the system will consider the current turn "
    "complete and proceed to the next turn or task."
)
_AVAILABLE_TOOLS = "Here is a list of functions in {format} format that you can invoke.\n{functions}\n"
_TEMPLATE = "{persona}{task}\n\n{tool_call_format}\n\n{multiturn_behavior}\n\n{available_tools}"
_PYTHON_HINT = " Note that the provided function is in Python 3 syntax."


def with_language_hint(functions: list[dict]) -> list[dict]:
    """`_func_doc_language_specific_pre_processing` for a Python category: the hint on each description."""
    out = copy.deepcopy(functions)
    for item in out:
        item["description"] = item["description"] + _PYTHON_HINT
    return out


def system_prompt(functions: list[dict]) -> str:
    """`formulate_system_prompt` with the default format, on already hinted function docs."""
    return _TEMPLATE.format(
        persona=_PERSONA,
        task=_TASK,
        tool_call_format=_TOOL_CALL_NO_TAG.format(output_format=_OUTPUT_FORMAT, param_types=""),
        multiturn_behavior=_MULTITURN,
        available_tools=_AVAILABLE_TOOLS.format(format="json", functions=json.dumps(functions, indent=4)),
    )


def messages(entry: dict) -> list[dict]:
    """The chat for one item: the system prompt, then the item's first-turn messages."""
    turn = copy.deepcopy(entry["question"][0])
    sp = system_prompt(with_language_hint(entry["function"]))
    if turn and turn[0]["role"] == "system":
        turn[0]["content"] = sp + "\n\n" + turn[0]["content"]
        return turn
    return [{"role": "system", "content": sp}, *turn]


# ---- decoding (bfcl_eval/model_handler/utils.py) -----------------------------------------------------


def _resolve_call(elem) -> dict:
    parts, f = [], elem.func
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return {".".join(reversed(parts)): {kw.arg: _resolve_value(kw.value) for kw in elem.keywords}}


def _resolve_value(value):
    if isinstance(value, ast.Constant):
        return "..." if value.value is Ellipsis else value.value
    if isinstance(value, ast.UnaryOp):
        return -value.operand.value
    if isinstance(value, ast.List):
        return [_resolve_value(v) for v in value.elts]
    if isinstance(value, ast.Dict):
        return {_resolve_value(k): _resolve_value(v) for k, v in zip(value.keys, value.values, strict=True)}
    if isinstance(value, ast.BinOp):
        # the official decoder eval()s the expression; only numeric arithmetic is evaluated here
        return _arith(value)
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Call):
        return ast.unparse(value) if not value.keywords else _resolve_call(value)
    if isinstance(value, ast.Tuple):
        return tuple(_resolve_value(v) for v in value.elts)
    if isinstance(value, ast.Subscript):
        return ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    raise ValueError(f"Unsupported AST type: {type(value).__name__}")


_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}


def _arith(node):
    """Numeric arithmetic on literals, as eval() would compute it; anything else is a decode failure."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _arith(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _arith(node.left), _arith(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > 64:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](a, b)
    raise ValueError(f"unsupported expression {ast.unparse(node)!r}")


def is_call_list(decoded) -> bool:
    """`is_function_calling_format_output`: a list of {name: {param: value}} (an empty list included)."""
    return isinstance(decoded, list) and all(
        isinstance(d, dict) and len(d) == 1 and isinstance(next(iter(d.values())), dict) for d in decoded
    )


def decode(text: str) -> list[dict]:
    """`default_decode_ast_prompting` for Python output; raises on anything that is not a call list."""
    result = text.strip("`\n ")
    if not result.startswith("["):
        result = "[" + result
    if not result.endswith("]"):
        result = result + "]"
    parsed = ast.parse(result.strip().strip("'"), mode="eval")
    if isinstance(parsed.body, ast.Call):
        return [_resolve_call(parsed.body)]
    out = []
    for elem in parsed.body.elts:
        if not isinstance(elem, ast.Call):
            raise ValueError("not a function call")
        out.append(_resolve_call(elem))
    return out


# ---- AST checker (bfcl_eval/eval_checker/ast_eval/ast_checker.py), Python path ------------------------

PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}
PYTHON_NESTED_TYPE_CHECK_LIST = ["array", "tuple"]


def _answer_type(possible_answer: list):
    for answer in possible_answer:
        if answer != "":
            return type(answer)
    return None


def _type_checker(param, value, possible_answer, expected_desc, expected_type, nested_type):
    result = {"valid": True, "error": [], "is_variable": False, "error_type": "type_error:simple"}
    is_variable = False
    pa_type = _answer_type(possible_answer)
    if pa_type is not None and pa_type != expected_type:
        is_variable = True
    if type(value) is expected_type:
        if nested_type is None:
            result["is_variable"] = is_variable
            return result
        for item in possible_answer:
            flag = True
            if type(item) is list:
                for v in value:
                    if not _type_checker(param, v, item, str(nested_type), nested_type, None)["valid"]:
                        flag = False
                        break
            if flag:
                return {"valid": True, "error": [], "is_variable": is_variable}
        result["valid"] = False
        result["error"] = [f"Nested type checking failed for parameter {param!r}."]
        result["error_type"] = "type_error:nested"
    pa_type = _answer_type(possible_answer)
    if pa_type is not None and type(value) is pa_type:
        result["is_variable"] = True
        return result
    result["valid"] = False
    result["error"].append(f"Incorrect type for parameter {param!r}. Expected type {expected_desc}.")
    result["error_type"] = "type_error:simple"
    return result


def standardize_string(s: str) -> str:
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", s).lower().replace("'", '"')


def _string_checker(param, value, possible_answer):
    std = [standardize_string(a) for a in possible_answer if type(a) is str]
    if standardize_string(value) not in std:
        return {
            "valid": False,
            "error": [f"Invalid value for {param!r}."],
            "error_type": "value_error:string",
        }
    return {"valid": True, "error": []}


def _list_checker(param, value, possible_answer):
    out = [standardize_string(v) if type(v) is str else v for v in list(value)]
    std = [[standardize_string(x) if type(x) is str else x for x in a] for a in possible_answer]
    if out not in std:
        return {
            "valid": False,
            "error": [f"Invalid value for {param!r}."],
            "error_type": "value_error:list/tuple",
        }
    return {"valid": True, "error": []}


def _dict_checker(param, value, possible_answers):
    result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
    for pa in possible_answers:
        if pa == "":
            continue
        result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
        flag = True
        for key, v in value.items():
            if key not in pa:
                result.update(error=[f"Unexpected dict key {key!r}."], error_type="value_error:dict_key")
                flag = False
                break
            sv = standardize_string(v) if type(v) is str else v
            std = [standardize_string(x) if type(x) is str else x for x in pa[key]]
            if sv not in std:
                result.update(error=[f"Invalid value for {key!r}."], error_type="value_error:dict_value")
                flag = False
                break
        for key, v in pa.items():
            if key not in value and "" not in v:
                result.update(error=[f"Missing dict key {key!r}."], error_type="value_error:dict_key")
                flag = False
                break
        if flag:
            return {"valid": True, "error": []}
    return result


def _list_dict_checker(param, value, possible_answers):
    result = {"valid": False, "error": [], "error_type": "list_dict_checker:unclear"}
    for pa in possible_answers:
        flag = True
        if len(value) != len(pa):
            result = {
                "valid": False,
                "error": ["Wrong number of dictionaries."],
                "error_type": "value_error:list_dict_count",
            }
            continue
        for i in range(len(value)):
            result = _dict_checker(param, value[i], [pa[i]])
            if not result["valid"]:
                flag = False
                break
        if flag:
            return {"valid": True, "error": []}
    return result


def simple_function_checker(func_description: dict, model_output: dict, possible_answer: dict) -> dict:
    possible_answer = list(possible_answer.values())[0]
    func_name = func_description["name"]
    param_details = func_description["parameters"]["properties"]
    required = func_description["parameters"]["required"]
    result = {"valid": True, "error": [], "error_type": "simple_function_checker:unclear"}
    if func_name not in model_output:
        return {
            "valid": False,
            "error": [f"Function {func_name!r} not found."],
            "error_type": "simple_function_checker:wrong_func_name",
        }
    params = model_output[func_name]
    for p in required:
        if p not in params:
            return {
                "valid": False,
                "error": [f"Missing required parameter {p!r}."],
                "error_type": "simple_function_checker:missing_required",
            }
    for p, value in params.items():
        if p not in param_details or p not in possible_answer:
            return {
                "valid": False,
                "error": [f"Unexpected parameter {p!r}."],
                "error_type": "simple_function_checker:unexpected_param",
            }
        desc = param_details[p]["type"]
        expected = PYTHON_TYPE_MAPPING[desc]
        nested = (
            PYTHON_TYPE_MAPPING[param_details[p]["items"]["type"]]
            if desc in PYTHON_NESTED_TYPE_CHECK_LIST
            else None
        )
        if desc == "tuple" and type(value) is tuple:
            value = list(value)
        if desc == "float" and type(value) is int:
            value = float(value)
        tc = _type_checker(p, value, possible_answer[p], desc, expected, nested)
        if not tc["valid"]:
            return tc
        if not tc["is_variable"]:
            if expected is dict:
                r = _dict_checker(p, value, possible_answer[p])
            elif expected is list and nested is dict:
                r = _list_dict_checker(p, value, possible_answer[p])
            elif expected is str:
                r = _string_checker(p, value, possible_answer[p])
            elif expected is list:
                r = _list_checker(p, value, possible_answer[p])
            else:
                r = None
            if r is not None:
                if not r["valid"]:
                    return r
                continue
        if value not in possible_answer[p]:
            return {
                "valid": False,
                "error": [f"Invalid value for {p!r}."],
                "error_type": "value_error:others",
            }
    for p in possible_answer:
        if p not in params and "" not in possible_answer[p]:
            return {
                "valid": False,
                "error": [f"Optional parameter {p!r} not provided."],
                "error_type": "simple_function_checker:missing_optional",
            }
    return result


def ast_check(
    category: str, functions: list[dict], model_output: list[dict], possible_answer: list[dict]
) -> dict:
    """`ast_checker` for the simple and multiple categories (Python)."""
    if category == "multiple":
        if len(model_output) != len(possible_answer):
            return {
                "valid": False,
                "error": ["Wrong number of functions."],
                "error_type": "multiple_function_checker:wrong_count",
            }
        name = list(possible_answer[0].keys())[0]
        desc = next((f for f in functions if f["name"] == name), None)
        return simple_function_checker(desc, model_output[0], possible_answer[0])
    if category != "simple":
        raise ValueError(f"category {category!r} is not in {CATEGORIES}")
    if len(model_output) != 1:
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "simple_function_checker:wrong_count",
        }
    return simple_function_checker(functions[0], model_output[0], possible_answer[0])


def score(entry: dict, answer: dict, text: str, category: str) -> dict:
    """{"valid", "error_type", "decoded"} for one model output (decoder failures are their own error type)."""
    try:
        decoded = decode(text)
    except Exception as e:  # noqa: BLE001 - any decode failure is the official "ast_decoder:decoder_failed"
        return {
            "valid": False,
            "error_type": "ast_decoder:decoder_failed",
            "decoded": None,
            "decode_error": str(e)[:200],
        }
    if not is_call_list(decoded):
        return {
            "valid": False,
            "error_type": "ast_decoder:decoder_wrong_output_format",
            "decoded": repr(decoded)[:500],
        }
    try:
        res = ast_check(category, entry["function"], decoded, answer["ground_truth"])
    except Exception as e:  # noqa: BLE001 - a checker crash on odd output counts as invalid, recorded
        return {
            "valid": False,
            "error_type": "ast_checker:crash",
            "decoded": repr(decoded)[:500],
            "decode_error": str(e)[:200],
        }
    return {
        "valid": bool(res["valid"]),
        "error_type": None if res["valid"] else res.get("error_type"),
        "decoded": repr(decoded)[:500],
    }


# ---- metrics -------------------------------------------------------------------------------------


def accuracy(rows: list[dict], replicates: int, seed: int) -> dict:
    """Accuracy per category and overall (items pooled), with a percentile bootstrap over items."""
    out = {}
    rng = np.random.default_rng(seed)
    for cat in (*CATEGORIES, "all"):
        rs = [r for r in rows if cat == "all" or r["category"] == cat]
        if not rs:
            continue
        v = np.array([float(r["valid"]) for r in rs])
        boot = v[rng.integers(0, len(v), (replicates, len(v)))].mean(axis=1)
        out[cat] = {
            "items": len(v),
            "mean": float(v.mean()),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        }
    return out


def paired_difference(
    arm: list[dict], base: list[dict], replicates: int, seed: int, threshold: float
) -> dict:
    """arm − base accuracy on the same items, paired bootstrap; a regression is a drop > threshold
    whose 95% CI excludes 0 (docs/prereg/phase4.md decision 4)."""
    b = {r["id"]: float(r["valid"]) for r in base}
    a = {r["id"]: float(r["valid"]) for r in arm}
    if set(a) != set(b):
        raise ValueError("arm and base runs cover different BFCL items")
    ids = sorted(a)
    d = np.array([a[i] - b[i] for i in ids])
    boot = d[np.random.default_rng(seed).integers(0, len(d), (replicates, len(d)))].mean(axis=1)
    lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    mean = float(d.mean())
    return {"items": len(d), "mean": mean, "ci95": [lo, hi], "regression": mean < -threshold and hi < 0}
