from sympy.logic.boolalg import And, Or, Not, Implies, Equivalent, to_cnf
from sympy import symbols

def example_sympy_logic_demo():
    """
    演示如何使用 Sympy 进行逻辑表达式构造、合取范式转换等。
    """
    # 创建3个命题变量 P, Q, R
    P, Q, R = symbols('P Q R')

    # 构造一个示例表达式: (P ∧ Q) => R
    expr = Implies(And(P, Q), R)
    print("[Sympy Demo] Original Expr:", expr)

    # 转换成合取范式
    cnf_expr = to_cnf(expr, simplify=True)
    print("[Sympy Demo] CNF Form:", cnf_expr)

    # 使用逻辑化简功能
    simplified_expr = expr.simplify()
    print("[Sympy Demo] Simplified Logic:", simplified_expr)

    # 可扩展功能：检查表达式的可满足性
    # satisfiability(cnf_expr) -> 返回满足该表达式的赋值组合。

def build_sympy_expressions_from_triples(triples):
    """
    将物理教材中的三元组映射为 Sympy 逻辑表达式。

    例如：三元组 (力, "causes", 加速度) -> Implies(力, 加速度)
          三元组 (牛顿第二定律, "explains", F=ma) -> Equivalent(牛顿第二定律, F=ma)
    """
    # 收集去重的实体
    unique_entities = set()
    for t in triples:
        unique_entities.add(t["head"])
        unique_entities.add(t["tail"])

    # 为实体创建符号映射
    entity_to_symbol = {}
    entity_symbols = symbols(' '.join([f'X{i}' for i in range(len(unique_entities))]))
    for i, ent in enumerate(unique_entities):
        entity_to_symbol[ent] = entity_symbols[i]

    # 遍历三元组，构造逻辑表达式
    expr_list = []
    for t in triples:
        head_sym = entity_to_symbol[t["head"]]
        rel = t["relation"].lower().strip()
        tail_sym = entity_to_symbol[t["tail"]]

        if rel in ["implies", "推断", "=>", "causes", "导致"]:
            # 因果或推导关系
            expr = Implies(head_sym, tail_sym)
        elif rel in ["and", "并且"]:
            # 同时发生的现象
            expr = And(head_sym, tail_sym)
        elif rel in ["or", "或者"]:
            # 可能性关系
            expr = Or(head_sym, tail_sym)
        elif rel in ["not", "非"]:
            # 否定关系
            expr = Not(head_sym)
        elif rel in ["explains", "解释", "等价于"]:
            # 定律与公式的解释关系
            expr = Equivalent(head_sym, tail_sym)
        elif rel in ["acts_on", "作用于"]:
            # 力对物体的作用关系
            expr = Implies(head_sym, tail_sym)
        elif rel in ["depends_on", "依赖于"]:
            # 条件依赖关系
            expr = Implies(tail_sym, head_sym)
        else:
            # 默认处理为 implies
            expr = Implies(head_sym, tail_sym)

        expr_list.append(expr)

    return expr_list, entity_to_symbol
