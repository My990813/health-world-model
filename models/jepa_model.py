"""
JEPA 世界模型层 (Layer 2) - 纯 NumPy MLP 实现
=============================================
Joint-Embedding Predictive Architecture 用于人体动力学建模

使用 MLP + 完整反向传播（CPU 友好）
训练策略：
  Phase 1: 自监督预训练 - 随机 mask 多天，在隐空间预测
  Phase 2: 监督微调 - 用体重真实值微调 decoder
"""

import numpy as np
import json
import os
import pickle
from typing import Optional, Tuple, List, Dict

STATS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "jepa_stats.json")
MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "jepa_model.pkl")

# ===================== 工具函数 =====================

def relu(x): return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(np.float32)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -20, 20)))
def tanh(x): return np.tanh(x)

def normalize(x, mean, std):
    return (x - mean) / (std + 1e-8)

def denormalize(x, mean, std):
    return x * (std + 1e-8) + mean

def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)

def cosine_loss(pred, target):
    """JEPA cosine embedding loss"""
    p_n = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
    t_n = target / (np.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
    return np.mean(1 - np.sum(p_n * t_n, axis=-1))


# ===================== MLP 模块 (完整反向传播) =====================

class MLP:
    """多层感知机，支持完整反向传播"""
    def __init__(self, dims: List[int], seed: int = 42):
        """
        dims: [input_dim, hidden1, hidden2, ..., output_dim]
        """
        self.layers = []
        self.n_layers = len(dims) - 1
        rng = np.random.RandomState(seed)

        for i in range(self.n_layers):
            W = (rng.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])).astype(np.float32)
            b = np.zeros(dims[i+1], dtype=np.float32)
            self.layers.append({'W': W, 'b': b})

        self.cache = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.cache = []
        h = x
        for i, layer in enumerate(self.layers):
            self.cache.append({'x': h})
            z = h @ layer['W'] + layer['b']
            if i < self.n_layers - 1:
                h = relu(z)
            else:
                h = z  # 输出层不加激活
            self.cache.append({'z': z, 'out': h})
        self.cache.append({'x': h})  # 最后一个输入
        return h

    def backward(self, grad: np.ndarray, lr: float = 1e-3, l2: float = 1e-4) -> np.ndarray:
        """反向传播"""
        for i in range(self.n_layers - 1, -1, -1):
            layer = self.layers[i]
            # cache 布局: [x0, z0+out0, x1, z1+out1, ..., xn]
            c_input = self.cache[i * 2]  # 该层的输入
            c_output = self.cache[i * 2 + 1]  # 该层的输出

            if i == self.n_layers - 1:
                dz = grad
            else:
                dz = grad * relu_grad(c_output['z'])

            layer['W'] -= lr * (c_input['x'].T @ dz / dz.shape[0] + l2 * layer['W'])
            layer['b'] -= lr * np.mean(dz, axis=0)

            if i > 0:
                grad = dz @ layer['W'].T

        return grad

    def parameters(self):
        count = 0
        for l in self.layers:
            count += l['W'].size + l['b'].size
        return count


# ===================== 时序编码器 =====================

class SequenceEncoder:
    """
    将时序输入展平后编码
    输入: (batch, seq_len, feat_dim) → 展平 (batch, seq_len*feat_dim) → MLP → latent
    """
    def __init__(self, seq_len: int, feat_dim: int, latent_dim: int,
                 hidden_dim: int = 128, seed: int = 42):
        flat_dim = seq_len * feat_dim
        self.encoder = MLP([flat_dim, hidden_dim, 64, latent_dim], seed=seed)
        self.seq_len = seq_len
        self.feat_dim = feat_dim

    def encode(self, x: np.ndarray) -> np.ndarray:
        """(batch, seq, feat) → (batch, latent)"""
        batch = x.shape[0]
        flat = x.reshape(batch, -1)
        return self.encoder.forward(flat)

    def encode_windows(self, x: np.ndarray) -> np.ndarray:
        """编码每个时间步的局部窗口，返回 (batch, seq, latent)"""
        batch, seq_len, feat_dim = x.shape
        latents = []
        window = self.seq_len

        for t in range(seq_len):
            if t < window:
                # padding
                pad = np.zeros((batch, window - t, feat_dim), dtype=np.float32)
                x_window = np.concatenate([pad, x[:, :t+1]], axis=1)
            else:
                x_window = x[:, t-window+1:t+1]
            z = self.encode(x_window)
            latents.append(z)

        return np.stack(latents, axis=1)


# ===================== JEPA 世界模型 =====================

class JEPAWorldModel:
    """
    JEPA 世界模型

    架构:
      输入序列 (14天 × 8特征) → SequenceEncoder → 隐状态 (32维)
      Predictor: 已知隐状态 → 预测被mask隐状态
      Decoder: 隐状态 → [体重, 代谢率]

    训练:
      1. 随机 mask 几天数据
      2. 用可见天编码为 latent → 预测被 mask 天的 latent
      3. Cosine loss in latent space (JEPA 风格)
      4. 额外: 用真实体重微调 decoder
    """

    def __init__(
        self,
        input_dim: int = 8,
        latent_dim: int = 32,
        context_len: int = 14,
        pred_horizon: int = 7,
        seed: int = 42
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.context_len = context_len
        self.pred_horizon = pred_horizon

        # Encoder: (seq*feat) → latent
        self.encoder = SequenceEncoder(
            seq_len=context_len, feat_dim=input_dim,
            latent_dim=latent_dim, hidden_dim=128, seed=seed
        )

        # Day-level encoder: 单天特征 → latent (用于逐天预测)
        self.day_encoder = MLP([input_dim, 64, latent_dim], seed=seed + 10)

        # Predictor: latent(history) + diet(future) → latent(future)
        self.predictor = MLP(
            [latent_dim + input_dim, 96, 64, latent_dim],
            seed=seed + 20
        )

        # Decoder: latent → [weight_delta, metabolic_rate]
        self.decoder = MLP([latent_dim, 48, 16, 2], seed=seed + 30)

        # 统计量
        self.feat_mean = np.zeros(input_dim, dtype=np.float32)
        self.feat_std = np.ones(input_dim, dtype=np.float32)
        self.weight_mean = np.array([70.0], dtype=np.float32)
        self.weight_std = np.array([5.0], dtype=np.float32)

        self.training_losses = []

    def encode_context(self, x: np.ndarray) -> np.ndarray:
        """编码完整上下文 → 全局 latent"""
        return self.encoder.encode(x)  # (batch, latent)

    def encode_day(self, x_day: np.ndarray) -> np.ndarray:
        """编码单天特征 → latent"""
        return self.day_encoder.forward(x_day)  # (batch, latent)

    def predict_day(self, z_context: np.ndarray, x_future: np.ndarray) -> np.ndarray:
        """
        预测单天的隐状态
        z_context: (batch, latent) - 上下文编码
        x_future: (batch, input_dim) - 未来一天的饮食
        returns: (batch, latent)
        """
        inp = np.concatenate([z_context, x_future], axis=-1)
        return self.predictor.forward(inp)

    def decode_state(self, z: np.ndarray) -> np.ndarray:
        """latent → [weight_delta, metabolic_rate]"""
        return self.decoder.forward(z)  # (batch, 2)

    def simulate_future(self, history: np.ndarray, diet_plan: np.ndarray) -> Dict:
        """
        虚拟试餐模拟器

        history: (1, 14, 8) 过去14天
        diet_plan: (1, N, 8) 未来N天饮食方案
        """
        batch = history.shape[0]
        future_len = diet_plan.shape[1]

        # 编码历史上下文
        z_ctx = self.encode_context(history)  # (1, latent)

        weight_preds = []
        metabolic_preds = []
        current_weight = denormalize(history[0, -1, 0:1], self.weight_mean, self.weight_std) if self.weight_mean[0] != 0 else 75.0

        for t in range(future_len):
            x_t = diet_plan[:, t, :]  # (1, 8)
            z_pred = self.predict_day(z_ctx, x_t)
            decoded = self.decode_state(z_pred)  # (1, 2)

            # decoded[0] = weight_delta (归一化的体重变化)
            # decoded[1] = metabolic_rate (归一化的代谢率)
            delta = decoded[0, 0] * self.weight_std[0] * 0.1  # 缩放因子
            new_weight = current_weight + delta
            current_weight = new_weight

            weight_preds.append(round(float(new_weight), 2))

            # 代谢率反归一化
            meta = denormalize(decoded[0:1, 1:2], np.array([1700.0]), np.array([200.0]))
            metabolic_preds.append(round(float(meta[0, 0]), 1))

            # 更新上下文（将预测结果纳入）
            z_ctx = z_pred

        return {
            'weight_pred': weight_preds,
            'metabolic_pred': metabolic_preds,
            'diet_plan': diet_plan[0, :, 0].tolist() if diet_plan.shape[-1] > 0 else [],
        }

    def train_epoch(self, x_sequences: np.ndarray, weight_targets: np.ndarray,
                     mask_ratio: float = 0.3, lr: float = 2e-3) -> Dict:
        """
        一个 epoch 的训练

        x_sequences: (N, seq_len, feat_dim)
        weight_targets: (N, seq_len)
        """
        N = x_sequences.shape[0]
        seq_len = x_sequences.shape[1]
        indices = np.random.permutation(N)

        total_loss = 0
        n_batches = 0
        batch_size = 16

        for start in range(0, N, batch_size):
            bi = indices[start:start + batch_size]
            x_batch = x_sequences[bi]
            w_batch = weight_targets[bi]

            # === JEPA 自监督目标 ===
            # 随机选择 mask 起始点
            mask_start = np.random.randint(seq_len - self.pred_horizon, size=len(bi))
            loss_batch = 0

            for i in range(len(bi)):
                ms = mask_start[i]

                # 可见历史（pad到seq_len）
                x_vis = x_batch[i:i+1, :ms+1]
                pad_len = seq_len - x_vis.shape[1]
                if pad_len > 0:
                    pad = np.zeros((1, pad_len, x_vis.shape[2]), dtype=np.float32)
                    x_visible = np.concatenate([pad, x_vis], axis=1)
                else:
                    x_visible = x_vis[:, -seq_len:]
                z_ctx = self.encode_context(x_visible)

                # 逐天预测未来
                for t in range(self.pred_horizon):
                    future_day = ms + 1 + t
                    if future_day >= seq_len:
                        break

                    x_future = x_batch[i:i+1, future_day]
                    z_pred = self.predict_day(z_ctx, x_future)

                    # 用该天的编码作为 target
                    z_target = self.encode_day(x_future)

                    # Cosine loss
                    cos = np.sum(z_pred * z_target) / (
                        np.linalg.norm(z_pred) * np.linalg.norm(z_target) + 1e-8
                    )
                    loss_batch += (1 - cos)

                    # Decoder 监督: 体重变化
                    if future_day > 0:
                        weight_delta = w_batch[i, future_day] - w_batch[i, future_day - 1]
                        delta_norm = normalize(
                            np.array([[weight_delta]]),
                            np.array([0.0]),
                            self.weight_std
                        )
                        decoded_full = self.decode_state(z_pred)  # (1, 2)
                        # 只对体重变化计算loss，代谢率部分grad为0
                        grad_dec = np.zeros_like(decoded_full)
                        grad_dec[:, 0:1] = 0.5 * (decoded_full[:, 0:1] - delta_norm) / decoded_full.shape[0]
                        self.decoder.backward(grad_dec, lr)

                    # Predictor 反向传播
                    grad_pred = -(z_target - z_pred * cos) / (
                        np.linalg.norm(z_pred) * np.linalg.norm(z_target) + 1e-8
                    )
                    self.predictor.backward(grad_pred, lr)

                    # Day encoder 反向传播
                    self.day_encoder.backward(grad_pred, lr * 0.5)

                    # 更新上下文
                    z_ctx = z_pred

            total_loss += loss_batch / max(self.pred_horizon, 1)
            n_batches += 1

        return {'loss': total_loss / max(n_batches, 1)}


def train_jepa(logs: List[Dict], epochs: int = 100, seq_len: int = 14,
               lr: float = 2e-3, mask_ratio: float = 0.3) -> JEPAWorldModel:
    """训练 JEPA 世界模型"""
    print("=" * 60)
    print("JEPA 世界模型训练 (MLP + Full Backprop)")
    print("=" * 60)

    # 提取序列
    features, weights = _extract_sequences(logs, seq_len)
    print(f"训练样本数: {features.shape[0]}")

    # 计算统计量
    feat_flat = features.reshape(-1, features.shape[-1])
    weight_flat = weights.reshape(-1)
    feat_mean = np.mean(feat_flat, axis=0).astype(np.float32)
    feat_std = np.std(feat_flat, axis=0).astype(np.float32)
    weight_mean = np.mean(weight_flat)
    weight_std = np.std(weight_flat)

    # 保存统计量
    stats = {
        'feat_mean': feat_mean.tolist(),
        'feat_std': feat_std.tolist(),
        'weight_mean': float(weight_mean),
        'weight_std': float(weight_std),
    }
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

    # 归一化
    feat_norm = normalize(features, feat_mean, feat_std)
    weight_norm = normalize(weights, np.array([weight_mean]), np.array([weight_std]))

    # 初始化模型
    model = JEPAWorldModel(
        input_dim=features.shape[-1],
        latent_dim=32,
        context_len=seq_len,
        pred_horizon=7
    )
    model.feat_mean = feat_mean
    model.feat_std = feat_std
    model.weight_mean = np.array([weight_mean], dtype=np.float32)
    model.weight_std = np.array([weight_std], dtype=np.float32)

    n_params = sum(
        mlp.parameters() for mlp in [model.encoder.encoder, model.day_encoder,
                                      model.predictor, model.decoder]
    )
    print(f"模型参数量: {n_params}")
    print(f"开始训练 {epochs} epochs")
    print("-" * 60)

    for epoch in range(epochs):
        result = model.train_epoch(feat_norm, weight_norm, mask_ratio, lr)
        model.training_losses.append(result['loss'])

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {result['loss']:.4f}")

    print("-" * 60)
    print(f"训练完成! 最终 Loss: {model.training_losses[-1]:.4f}")

    # 保存模型
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'model': model,
            'stats': stats,
            'losses': model.training_losses
        }, f)
    print(f"模型已保存: {MODEL_FILE}")

    return model


def load_jepa_model() -> Tuple[Optional[JEPAWorldModel], Optional[Dict]]:
    """加载已训练的模型"""
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data.get('stats', {})
    return None, None


def _extract_sequences(logs: List[Dict], seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """从日志提取序列"""
    if len(logs) < seq_len:
        raise ValueError(f"需要至少 {seq_len} 天数据")

    n_samples = len(logs) - seq_len + 1
    features = []
    weights = []

    for i in range(n_samples):
        window = logs[i:i + seq_len]
        feat_window = []
        weight_window = []

        for day in window:
            total_cal = (day.get('breakfast_cal', 0) + day.get('lunch_cal', 0) +
                         day.get('dinner_cal', 0) + day.get('snack_cal', 0))
            feat_window.append([
                total_cal,
                day.get('protein_g', 0),
                day.get('carbs_g', 0),
                day.get('fat_g', 0),
                day.get('exercise_min', 0),
                day.get('sleep_hours', 7),
                day.get('meal_rating', 3),
                day.get('water_ml', 2000) / 1000.0,
            ])
            weight_window.append(day.get('weight', 70))

        features.append(feat_window)
        weights.append(weight_window)

    return np.array(features, dtype=np.float32), np.array(weights, dtype=np.float32)


def predict_weight_trend(logs: List[Dict], diet_plan: List[Dict],
                          model: Optional[JEPAWorldModel] = None) -> Dict:
    """虚拟试餐模拟器接口"""
    if model is None:
        model, stats = load_jepa_model()

    if model is None or len(logs) < model.context_len:
        return simple_physics_predict(logs, diet_plan)

    recent = logs[-model.context_len:]
    history = np.zeros((1, model.context_len, model.input_dim), dtype=np.float32)
    for i, day in enumerate(recent):
        total_cal = (day.get('breakfast_cal', 0) + day.get('lunch_cal', 0) +
                     day.get('dinner_cal', 0) + day.get('snack_cal', 0))
        history[0, i] = [
            total_cal, day.get('protein_g', 0), day.get('carbs_g', 0),
            day.get('fat_g', 0), day.get('exercise_min', 0),
            day.get('sleep_hours', 7), day.get('meal_rating', 3),
            day.get('water_ml', 2000) / 1000.0
        ]

    history_norm = normalize(history, model.feat_mean, model.feat_std)

    future_len = len(diet_plan)
    diet_input = np.zeros((1, future_len, model.input_dim), dtype=np.float32)
    for i, day in enumerate(diet_plan):
        diet_input[0, i] = [
            day.get('total_cal', 1500), day.get('protein_g', 100),
            day.get('carbs_g', 150), day.get('fat_g', 50),
            day.get('exercise_min', 0), 7, 3, 2
        ]

    diet_norm = normalize(diet_input, model.feat_mean, model.feat_std)
    return model.simulate_future(history_norm, diet_norm)


def simple_physics_predict(logs: List[Dict], diet_plan: List[Dict]) -> Dict:
    """简单物理模型 fallback"""
    current_weight = logs[-1].get('weight', 75.0) if logs else 75.0

    bmr = 10 * current_weight + 6.25 * 175 - 5 * 28 + 5
    weight_preds = [current_weight]
    meta_preds = [bmr]

    for day in diet_plan:
        total_cal = day.get('total_cal', 1500)
        exercise = day.get('exercise_min', 0)
        tdee = (bmr * 1.2 + total_cal * 0.1) + exercise * 7
        deficit = total_cal - tdee
        new_w = weight_preds[-1] + deficit / 7700
        weight_preds.append(round(new_w, 2))
        new_bmr = 10 * new_w + 6.25 * 175 - 5 * 28 + 5
        meta_preds.append(round(new_bmr, 1))

    return {
        'weight_pred': weight_preds[1:],
        'metabolic_pred': meta_preds[1:],
        'diet_plan': [d.get('total_cal', 1500) for d in diet_plan],
        'method': 'simple_physics'
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.db_layer import get_all_logs

    logs = get_all_logs()
    print(f"总记录数: {len(logs)}")

    model = train_jepa(logs, epochs=100, seq_len=14, lr=2e-3)
