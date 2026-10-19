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
    tree = ast.parse(source)
    # f-string tokenization differs across Python minors; reject it explicitly
    # before constructing a fingerprint rather than silently changing configHash.
    if any(isinstance(node, ast.JoinedStr) for node in ast.walk(tree)):
        raise ValueError('POLICY_FSTRING_NOT_SUPPORTED')
    # AST validates syntax and locates docstrings; node fields never enter the hash.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                def position(line, byte_column):
                    return (line, len(lines[line - 1].encode('utf-8')[:byte_column].decode('utf-8')))
                spans.append((position(first.lineno, first.col_offset), position(first.end_lineno, first.end_col_offset)))
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.ENDMARKER, tokenize.ENCODING}
    result = []
    statement_has_code = False
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in ignored or any(start <= token.start and token.end <= end for start, end in spans):
            continue
        if token.type == tokenize.NEWLINE:
            if statement_has_code:
                result.append(['NEWLINE', ''])
            statement_has_code = False
            continue
        if token.type not in {tokenize.INDENT, tokenize.DEDENT}:
            statement_has_code = True
        value = '' if token.type in {tokenize.INDENT, tokenize.DEDENT} else token.string
        result.append([tokenize.tok_name[token.type], value])
    return result
