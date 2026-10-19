"""Policy source normalization without interpreter-specific AST serialization."""
from __future__ import annotations

import ast
import io
import textwrap
import tokenize


def normalized_policy_tokens(source):
    source = textwrap.dedent(source)
    lines = source.splitlines(keepends=True)
    spans = []
    # AST is used only to locate true docstrings; node fields never enter the hash.
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                def position(line, byte_column):
                    return (line, len(lines[line - 1].encode('utf-8')[:byte_column].decode('utf-8')))
                spans.append((position(first.lineno, first.col_offset), position(first.end_lineno, first.end_col_offset)))
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENDMARKER, tokenize.ENCODING}
    result = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in ignored or any(start <= token.start and token.end <= end for start, end in spans):
            continue
        value = '' if token.type in {tokenize.INDENT, tokenize.DEDENT} else token.string
        result.append([tokenize.tok_name[token.type], value])
    return result
