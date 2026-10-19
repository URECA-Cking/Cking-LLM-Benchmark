import ast
from src.recommendation.policy_fingerprint import normalized_policy_tokens


def test_comments_docstrings_and_layout_do_not_change_tokens():
    plain = "def policy(x):\n    return x + 3\n"
    documented = 'def policy( x ):\n    """정책 설명\n    문서 줄\n    """\n    # comment\n    return x+3 # another comment\n'
    assert normalized_policy_tokens(plain) == normalized_policy_tokens(documented)
    assert normalized_policy_tokens(plain) != normalized_policy_tokens(plain.replace('+ 3', '+ 4'))


def test_true_string_literals_and_indentation_remain_part_of_policy():
    first = 'def policy(x):\n    if x:\n        return "keep"\n    return "other"\n'
    assert normalized_policy_tokens(first) != normalized_policy_tokens(first.replace('"keep"', '"changed"'))
    assert normalized_policy_tokens(first) != normalized_policy_tokens(first.replace('    return "other"', '        return "other"'))


def test_parenthesized_docstring_and_unicode_columns():
    first = 'def policy():\n    ("설명" "추가")\n    value = "보존"\n    return value\n'
    second = 'def policy():\n    value = "보존"\n    return value\n'
    assert normalized_policy_tokens(first) == normalized_policy_tokens(second)


def test_interpreter_ast_fields_never_enter_fingerprint(monkeypatch):
    source = 'def policy():\n    """description"""\n    return 3\n'
    baseline = normalized_policy_tokens(source)
    parse = ast.parse
    def added_metadata(text):
        tree = parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                node.type_params = []
                node.future_interpreter_metadata = 'arbitrary'
        return tree
    monkeypatch.setattr(ast, 'parse', added_metadata)
    monkeypatch.setattr(ast, 'dump', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('AST serialization forbidden')))
    assert normalized_policy_tokens(source) == baseline
