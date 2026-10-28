import numpy as np
import statsmodels.api as sm


def run_causal_mediation(protocol_M, overreaction_vec, paralysis_vec, ttr_outcome, n_boot=2000):
    """
    结构方程中介分析 (SEM Decomposition via Bootstrap, Section V-E)
    M4 (Treatment=1) vs M5 (Control=0) 对 TTR 的间接影响分解
    """
    X = protocol_M  # Treatment: 1 for M4, 0 for M5
    M1 = overreaction_vec
    M2 = paralysis_vec
    Y = ttr_outcome

    # 1. 拟合中介方程: M1 = i1 + a1*X, M2 = i2 + a2*X
    a1 = sm.OLS(M1, sm.add_constant(X)).fit().params[1]
    a2 = sm.OLS(M2, sm.add_constant(X)).fit().params[1]

    # 2. 拟合产出方程: Y = i3 + c'*X + b1*M1 + b2*M2
    X_mat = np.column_stack((X, M1, M2))
    res_y = sm.OLS(Y, sm.add_constant(X_mat)).fit()
    b1 = res_y.params[2]
    b2 = res_y.params[3]

    indirect_m1 = a1 * b1  # Overreaction 效应 (论文约 0.84)
    indirect_m2 = a2 * b2  # Paralysis 效应 (论文约 0.53)
    total_indirect = indirect_m1 + indirect_m2

    return {
        "indirect_overreaction": indirect_m1,
        "indirect_paralysis": indirect_m2,
        "total_indirect": total_indirect,
        "explained_ratio": total_indirect / (np.mean(Y[X == 1]) - np.mean(Y[X == 0]))
    }