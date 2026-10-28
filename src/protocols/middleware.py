class ProtocolMiddleware:
    """
    Layer 2: 信息可见性管控治理中间件
    Regimes: M1 (Siloed), M2 (EDI), M3 (Delayed), M4 (Full Raw), M5 (Federated Knowledge)
    """
    def __init__(self, mode="M5", delay_tau=2):
        self.mode = mode
        self.delay_tau = delay_tau
        self.history_buffer = []

    def filter_telemetry(self, raw_iot_perceived, current_orders):
        self.history_buffer.append(raw_iot_perceived)

        if self.mode == "M1":
            # 孤立数据：只能看各自本地的 RFID，跨节点全不可见
            return {f"agent_{i}": {"local_rfid": raw_iot_perceived['rfid_stock'][i]} for i in range(5)}

        elif self.mode == "M2":
            # 传统 EDI：只能看直接上下游标量订货量，完全不广播 IoT
            return {"shared_orders": current_orders}

        elif self.mode == "M3":
            # 引入时间低通滤波器：延迟 tau 期广播完整 IoT
            idx = max(0, len(self.history_buffer) - 1 - self.delay_tau)
            return {"delayed_iot_stream": self.history_buffer[idx]}

        elif self.mode == "M4":
            # 朴素全透明：全量高频原始流直接全网广播 (CRP 触发源)
            return {"raw_iot_stream": raw_iot_perceived}

        elif self.mode == "M5":
            # 生产级联邦知识：提取高阶语义抽象，阻止原始噪声传播
            stocks = raw_iot_perceived['rfid_stock']
            temps = raw_iot_perceived['temp_reading']
            return {
                "federated_risk_summary": {
                    "upstream_stockout_risk": "HIGH" if (stocks[0] < 40 or stocks[1] < 30) else "NORMAL",
                    "cold_chain_anomaly": "CRITICAL" if any(t > 8.0 for t in temps) else "NORMAL",
                    "network_avg_stock": float(np.mean(stocks))
                }
            }