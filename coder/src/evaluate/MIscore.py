import radon.metrics
import radon.complexity
import radon.raw
import radon.visitors

def calculate_MI(code):
    # 计算维护指数（MI）
    mi_score = radon.metrics.mi_visit(code, True)

    # 计算圈复杂度（Cyclomatic Complexity）
    cc_result = radon.complexity.cc_visit(code)
    cc_score = sum(block.complexity for block in cc_result) / len(cc_result) if cc_result else 0

    # 计算原子复杂度（Halstead Metrics）
    halstead_result = radon.metrics.h_visit(code)
    he_score = halstead_result.total.volume

    # 计算源码行数（SLOC）
    raw_metrics = radon.raw.analyze(code)
    sloc_score = raw_metrics.sloc
    return mi_score, cc_score, he_score, sloc_score