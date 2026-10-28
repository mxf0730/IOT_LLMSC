import numpy as np


class ATWCMechanism:
    """
    Adaptive Trust-Weighted Coordination (ATWC) 治理契约实现
    """

    def __init__(self, num_agents=5, gamma=0.8, kappa=2.0, lam=0.5, beta=0.9, eta=10.0):
        self.gamma = gamma
        self.kappa = kappa
        self.lam = lam
        self.beta = beta
        self.eta = eta
        self.ewma_actions = np.zeros(num_agents)
        self.trust_matrix = np.ones((num_agents, num_agents))  # 信任权重向量矩阵

    def apply_signal_dampening(self, agent_idx, intended_action, historical_std=5.0):
        """
        D1: 信号自适应阻尼平抑 (Eq. 25-27)
        """
        prev_ewma = self.ewma_actions[agent_idx]
        self.ewma_actions[agent_idx] = self.gamma * prev_ewma + (1.0 - self.gamma) * intended_action

        dev = abs(intended_action - prev_ewma) / max(1e-3, historical_std)
        # 计算阻尼系数 alpha
        alpha = 1.0 / (1.0 + self.lam * max(0.0, dev - self.kappa))
        executed_action = alpha * intended_action + (1.0 - alpha) * self.ewma_actions[agent_idx]
        return executed_action

    def update_trust(self, agent_i, agent_j, predicted_val, ground_truth):
        """
        D2: 动态信任度矩阵更新 (Eq. 28)
        """
        error = abs(predicted_val - ground_truth)
        w_prev = self.trust_matrix[agent_i, agent_j]
        self.trust_matrix[agent_i, agent_j] = self.beta * w_prev + (1.0 - self.beta) * np.exp(-error / self.eta)

    def aggregate_trusted_signal(self, agent_i, local_signal, peer_signals, rho=0.6):
        """
        D2: 信任加权信号聚合 (Eq. 29)
        """
        weights = [self.trust_matrix[agent_i, j] for j in range(len(peer_signals))]
        sum_w = sum(weights) + 1e-6
        weighted_peer = sum(w * s for w, s in zip(weights, peer_signals)) / sum_w
        return rho * local_signal + (1.0 - rho) * weighted_peer