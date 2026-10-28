import numpy as np


class PhysicalSCMEnvironment:
    """
    Layer 0: 严格基于 OR-Gym (InvManagement-v1) 和 Sterman 啤酒游戏动力学的物理环境底座
    节点设定：两家供应商 (S1, S2)、中央制造商 (M)、两家零售商 (R1, R2)
    """

    def __init__(self, config):
        self.cfg = config
        self.T = config['simulation']['horizon']
        self.reset()

    def reset(self):
        self.t = 0
        # 节点初始物理库存：[S1, S2, M, R1, R2]
        self.I_true = np.array([100.0, 70.0, 60.0, 30.0, 30.0])
        # 在途管道队列 (Pipeline queues: stage x lead_time)
        self.pipeline_S_M = np.zeros((2, self.cfg['simulation']['lead_times']['s_to_m']))
        self.pipeline_M_R = np.zeros((2, self.cfg['simulation']['lead_times']['m_to_r']))
        self.backorders = np.zeros(2)  # R1, R2
        self.temp_excursion_counter = np.zeros(5)  # 跟踪冷链超温持续时间
        return self.get_true_state()

    def get_true_state(self):
        return {
            't': self.t,
            'I_true': self.I_true.copy(),
            'pipeline_S_M': self.pipeline_S_M.copy(),
            'pipeline_M_R': self.pipeline_M_R.copy(),
            'backorders': self.backorders.copy()
        }

    def step(self, actions, true_ambient_temps, ext_demand):
        """
        状态转移方程执行 (Eq. 10 & Eq. 13)
        actions: dict 包含各节点动作，如 order_R1, order_R2, prod_M, proc_S1, proc_S2
        """
        # 1. 物理计算冷链损耗 (Thermal Decay Kinetics, Eq. 4-5)
        spoilage_losses = np.zeros(5)
        for i in range(5):
            if true_ambient_temps[i] > self.cfg['iot_noise']['cold_chain']['t_crit']:
                self.temp_excursion_counter[i] += 1
            else:
                self.temp_excursion_counter[i] = 0

            if self.temp_excursion_counter[i] >= self.cfg['iot_noise']['cold_chain']['tau_tol']:
                t_diff = (true_ambient_temps[i] - self.cfg['iot_noise']['cold_chain']['t_crit']) / \
                         self.cfg['iot_noise']['cold_chain']['t_crit']
                decay_rate = self.cfg['iot_noise']['cold_chain']['alpha_spoil'] * max(0.0, t_diff)
                spoilage_losses[i] = min(self.I_true[i], decay_rate * self.I_true[i])

        # 物理库存先扣减变质数量
        self.I_true = np.maximum(0.0, self.I_true - spoilage_losses)

        # 2. 零售商需求与入库物理流
        arrived_at_R = self.pipeline_M_R[:, 0]
        self.I_true[3] += arrived_at_R[0]
        self.I_true[4] += arrived_at_R[1]

        # 履行客户需求
        fulfilled_R1 = min(self.I_true[3], ext_demand[0] + self.backorders[0])
        self.backorders[0] = (ext_demand[0] + self.backorders[0]) - fulfilled_R1
        self.I_true[3] -= fulfilled_R1

        fulfilled_R2 = min(self.I_true[4], ext_demand[1] + self.backorders[1])
        self.backorders[1] = (ext_demand[1] + self.backorders[1]) - fulfilled_R2
        self.I_true[4] -= fulfilled_R2

        # 3. 制造商生产与入库物理流
        arrived_at_M = self.pipeline_S_M[:, 0]
        # BOM 装配约束 (1:1 消耗 S1 和 S2 原料)
        usable_mat = min(arrived_at_M[0], arrived_at_M[1])
        self.I_true[2] += usable_mat

        # 推进管道队列 (Shift pipelines)
        self.pipeline_S_M[:, :-1] = self.pipeline_S_M[:, 1:]
        self.pipeline_S_M[:, -1] = [actions.get('proc_S1', 20), actions.get('proc_S2', 20)]

        self.pipeline_M_R[:, :-1] = self.pipeline_M_R[:, 1:]
        dispatch_M_R1 = min(self.I_true[2] * 0.5, actions.get('order_R1', 15))
        dispatch_M_R2 = min(self.I_true[2] * 0.5, actions.get('order_R2', 15))
        self.I_true[2] -= (dispatch_M_R1 + dispatch_M_R2)
        self.pipeline_M_R[:, -1] = [dispatch_M_R1, dispatch_M_R2]

        self.t += 1
        return self.get_true_state(), spoilage_losses