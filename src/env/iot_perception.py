import numpy as np
from scipy.stats import poisson

class IoTPerceptionLayer:
    """
    Layer 1: 真实的工业传感器信道退化转换模型 (Gilbert-Elliott Markov信道 + Poisson时延)
    """
    def __init__(self, config):
        self.cfg = config['iot_noise']
        # Gilbert-Elliott 状态 (0: Good 正常, 1: Bad 遮挡突发掉包)
        self.rfid_markov_state = np.zeros(5, dtype=int)
        self.last_reported_rfid = np.zeros(5)

    def transduce_sensors(self, true_state, ambient_temps):
        t = true_state['t']
        I_true = true_state['I_true']
        perceived = {'rfid_stock': [], 'gps_eta': [], 'temp_reading': []}

        for i in range(5):
            # 1. RFID Gilbert-Elliott 信道突发掉包 (Eq. 1-2)
            if self.rfid_markov_state[i] == 0:
                if np.random.rand() < self.cfg['rfid']['p_bg']:
                    self.rfid_markov_state[i] = 1
            else:
                if np.random.rand() < self.cfg['rfid']['p_gb']:
                    self.rfid_markov_state[i] = 0

            if self.rfid_markov_state[i] == 1:
                # 掉包：上报陈旧的历史数值 (IRI 现象)
                reported_stock = self.last_reported_rfid[i]
            else:
                # 正常：带高斯高频白噪声
                reported_stock = max(0.0, I_true[i] + np.random.normal(0, self.cfg['rfid']['sigma']))
                self.last_reported_rfid[i] = reported_stock
            perceived['rfid_stock'].append(reported_stock)

            # 2. GPS Telemetry 泊松时延抖动与盲区漂移 (Eq. 3)
            jitter = poisson.rvs(self.cfg['gps']['lambda_delay'])
            dead_reckon = np.random.normal(0, self.cfg['gps']['sigma_dead'])
            perceived['gps_eta'].append(2.0 + jitter + dead_reckon)

            # 3. 环境传感器：带观测精度的高斯温漂 (Eq. 4-5)
            perceived['temp_reading'].append(ambient_temps[i] + np.random.normal(0, 0.4))

        return perceived