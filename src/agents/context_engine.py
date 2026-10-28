import json


class HierarchicalContextEngine:
    """
    Layer 3 核心组件: 三层上下文管理引擎，彻底杜绝数据直接 dump 给大模型
    """

    def __init__(self, agent_role, capacity_limit=100.0, window_size=5):
        self.role = agent_role
        self.capacity_limit = capacity_limit
        self.window_size = window_size
        self.tier1_working_memory = {}
        self.tier2_episodic_buffer = []

    def update_and_build_prompt(self, current_obs, protocol_data, turn_history):
        # Tier 1: 刷新即时角色边界与合同约束
        self.tier1_working_memory = {
            "role": self.role,
            "capacity_limit": self.capacity_limit,
            "local_inventory": current_obs.get('local_rfid', 0.0)
        }
        # Tier 2: 滚动窗口捕获趋势 (提取 depletion rate)
        self.tier2_episodic_buffer.append(self.tier1_working_memory['local_inventory'])
        if len(self.tier2_episodic_buffer) > self.window_size:
            self.tier2_episodic_buffer.pop(0)

        burn_rate = 0.0
        if len(self.tier2_episodic_buffer) >= 2:
            burn_rate = self.tier2_episodic_buffer[-1] - self.tier2_episodic_buffer[0]

        # Tier 3: 语义总结器输出结构化 Prompt
        structured_prompt = {
            "ROLE_INVARIANTS": self.tier1_working_memory,
            "OPERATIONAL_TREND": {"inventory_delta_trend": round(burn_rate, 2)},
            "GOVERNED_SHARED_INFO": protocol_data,
            "CONVERSATION_CONTEXT": turn_history[-2:] if turn_history else []
        }
        return json.dumps(structured_prompt, indent=2)


class ConstraintGuardedAlignment:
    """
    物理约束对齐护栏：防止大模型幻觉与违背物料平衡
    """

    @staticmethod
    def validate_action(raw_action_val, min_val=0.0, max_val=120.0):
        try:
            val = float(raw_action_val)
            return max(min_val, min(val, max_val))
        except (ValueError, TypeError):
            return 20.0  # 异常安全兜底值