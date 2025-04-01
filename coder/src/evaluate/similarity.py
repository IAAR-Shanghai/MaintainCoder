import difflib
import ast

def code_sim(raw_code, new_code):
    """
    计算代码改动行数与比例。
    """
    matcher = difflib.SequenceMatcher(None, raw_code.splitlines(), new_code.splitlines())
    total_changes = sum(op[4]-op[3] for op in matcher.get_opcodes() if op[0] != 'equal')
    change_ratio = total_changes / len(raw_code.splitlines()) * 100
    return total_changes, change_ratio

def ast_sim(raw_code, new_code):
    """
    计算AST树的相似度。
    """
    tree1 = parse_code_to_ast(raw_code)
    tree2 = parse_code_to_ast(new_code)
    if tree1 and tree2:
        similarity = calculate_similarity(tree1, tree2)
        return similarity
    return None

def parse_code_to_ast(code):
    """
    解析代码并生成 AST。
    """
    try:
        tree = ast.parse(code)
        return tree
    except SyntaxError as e:
        print(f"Syntax error in code: {e}")
        return None

def calculate_similarity(tree1, tree2):
    """
    计算两棵 AST 树的相似性。
    """
    tree1_str = ast.dump(tree1)
    tree2_str = ast.dump(tree2)
    
    # 使用 difflib.SequenceMatcher 计算相似度
    matcher = difflib.SequenceMatcher(None, tree1_str, tree2_str)
    similarity = matcher.ratio()
    
    return similarity
