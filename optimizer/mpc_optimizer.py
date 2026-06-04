"""
MPC 滚动优化层 (Layer 3)
=========================
Model Predictive Control 用于多天营养规划

替换传统 Q-learning，使用显式优化：
  优化变量：未来7天每日目标热量、蛋白、碳水
  优化目标：
    ① 贴近周度理想减重速率 (0.3~0.5 kg/周)
    ② 热量不能过低 (<1000kcal)
    ③ 蛋白 ≥ 1.6g/kg 体重
    ④ 营养配比在健康区间
  约束：
    - 热量: [1000, TDEE * 1.0]
    - 蛋白: [1.6 * weight, 2.2 * weight] g
    - 碳水: [总热量*30%, 总热量*45%] 的 kcal 来自碳水
    - 脂肪: [总热量*20%, 总热量*35%] 的 kcal 来自脂肪
    - 单餐食材热量 ≤ 40% 单餐热量

使用 scipy.optimize.minimize (SLSQP) 求解
"""

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import json
import os
from typing import Dict, List, Optional, Tuple

MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "jepa_model.pkl")


class MPCOptimizer:
    """MPC 滚动优化器"""

    def __init__(
        self,
        horizon: int = 7,
        target_weight_loss_weekly: float = 0.4,  # kg/周
        current_weight: float = 75.0,
        height_cm: float = 175.0,
        age: int = 28,
        sex: str = 'male',
        activity_level: float = 1.2,  # 久坐
        min_cal: float = 1000,
        protein_per_kg: float = 1.6,
        seed: int = 42
    ):
        self.horizon = horizon
        self.target_loss_weekly = target_weight_loss_weekly
        self.current_weight = current_weight
        self.height_cm = height_cm
        self.age = age
        self.sex = sex
        self.activity_level = activity_level
        self.min_cal = min_cal
        self.protein_per_kg = protein_per_kg

        # 计算基础代谢率 (Mifflin-St Jeor)
        self.bmr = self._compute_bmr()
        self.tdee = self.bmr * activity_level

        # 每日目标热量缺口
        # 0.4 kg/周 ≈ 400g/7天 ≈ 57g/天 ≈ 57*7700/7 ≈ 626 kcal/天 缺口
        self.daily_deficit_target = (self.target_loss_weekly * 7700) / 7  # kcal

    def _compute_bmr(self) -> float:
        """Mifflin-St Jeor 公式"""
        if self.sex == 'male':
            return 10 * self.current_weight + 6.25 * self.height_cm - 5 * self.age + 5
        else:
            return 10 * self.current_weight + 6.25 * self.height_cm - 5 * self.age - 161

    def _compute_tdee_at_weight(self, weight: float) -> float:
        """给定体重的 TDEE"""
        if self.sex == 'male':
            bmr = 10 * weight + 6.25 * self.height_cm - 5 * self.age + 5
        else:
            bmr = 10 * weight + 6.25 * self.height_cm - 5 * self.age - 161
        return bmr * self.activity_level

    def objective(self, x: np.ndarray) -> float:
        """
        MPC 目标函数
        x: [cal_0, ..., cal_6, protein_0, ..., protein_6, carbs_0, ..., carbs_6]
        总共 horizon * 3 个变量
        """
        h = self.horizon
        cals = x[:h]
        proteins = x[h:2*h]
        carbs = x[2*h:3*h]

        loss = 0.0
        weight = self.current_weight

        for t in range(h):
            # 1. 减重速率目标（最小化与理想缺口的偏差）
            tdee_t = self._compute_tdee_at_weight(weight)
            deficit_t = cals[t] - tdee_t  # 正值=摄入>消耗=增重
            ideal_deficit = -self.daily_deficit_target  # 负值=热量缺口=减重
            loss += 2.0 * (deficit_t - ideal_deficit) ** 2 / (ideal_deficit ** 2 + 1)

            # 2. 蛋白质目标（鼓励高蛋白）
            protein_target = self.protein_per_kg * weight
            if proteins[t] < protein_target:
                loss += 5.0 * (protein_target - proteins[t]) ** 2 / (protein_target ** 2 + 1)

            # 3. 碳水偏好（适中碳水，鼓励在总热量30-40%范围内）
            carbs_kcal = carbs[t] * 4
            carbs_ratio = carbs_kcal / (cals[t] + 1)
            ideal_carbs_ratio = 0.35
            loss += 1.0 * (carbs_ratio - ideal_carbs_ratio) ** 2

            # 4. 脂肪约束（隐含：fat = (cal - protein*4 - carbs*4) / 9）
            fat_kcal = cals[t] - proteins[t] * 4 - carbs[t] * 4
            fat_ratio = fat_kcal / (cals[t] + 1)
            ideal_fat_ratio = 0.25
            loss += 1.0 * (fat_ratio - ideal_fat_ratio) ** 2

            # 5. 热量平滑性（避免剧烈波动）
            if t > 0:
                loss += 0.3 * (cals[t] - cals[t-1]) ** 2 / 10000

            # 6. 预测体重变化（正值=摄入>消耗=增重）
            weight += deficit_t / 7700

        return loss

    def optimize(self, current_weight: Optional[float] = None) -> Dict:
        """
        执行 MPC 滚动优化

        Returns:
          {
            'daily_plan': [
              {'cal': X, 'protein_g': Y, 'carbs_g': Z, 'fat_g': W},
              ...7天
            ],
            'predicted_weights': [w1, w2, ..., w7],
            'total_deficit': X,
            'target_deficit': Y
          }
        """
        if current_weight:
            self.current_weight = current_weight
            self.bmr = self._compute_bmr()
            self.tdee = self.bmr * self.activity_level

        h = self.horizon
        n_vars = h * 3  # cal, protein, carbs per day

        # 初始猜测：基于 TDEE - deficit
        target_cal = self.tdee - self.daily_deficit_target
        x0 = np.zeros(n_vars)
        x0[:h] = np.full(h, target_cal)  # 热量
        x0[h:2*h] = np.full(h, self.protein_per_kg * self.current_weight)  # 蛋白
        x0[2*h:3*h] = np.full(h, target_cal * 0.35 / 4)  # 碳水

        # 边界约束
        lb = np.zeros(n_vars)
        ub = np.zeros(n_vars)

        for t in range(h):
            tdee_t = self._compute_tdee_at_weight(
                self.current_weight - self.daily_deficit_target * t / 7700
            )

            # 热量: [1000, TDEE]
            lb[t] = self.min_cal
            ub[t] = tdee_t  # 不超过 TDEE（减脂）

            # 蛋白: [1.6g/kg, 2.5g/kg]
            weight_t = self.current_weight - self.daily_deficit_target * t / 7700
            lb[h + t] = self.protein_per_kg * weight_t
            ub[h + t] = 2.5 * weight_t

            # 碳水: 至少50g, 最多总热量45%
            lb[2*h + t] = 50
            ub[2*h + t] = tdee_t * 0.45 / 4

        bounds = Bounds(lb, ub)

        # 不等式约束: fat_kcal = cal - protein*4 - carbs*4 >= cal * 0.20
        # 即 cal - protein*4 - carbs*4 - cal*0.20 >= 0
        # 即 0.80*cal - 4*protein - 4*carbs >= 0

        def fat_min_constraint(x):
            h_local = self.horizon
            cals = x[:h_local]
            proteins = x[h_local:2*h_local]
            carbs = x[2*h_local:3*h_local]
            fat_kcal = cals - proteins * 4 - carbs * 4
            fat_ratio = fat_kcal / (cals + 1)
            return fat_ratio - 0.20  # >= 0

        def fat_max_constraint(x):
            h_local = self.horizon
            cals = x[:h_local]
            proteins = x[h_local:2*h_local]
            carbs = x[2*h_local:3*h_local]
            fat_kcal = cals - proteins * 4 - carbs * 4
            fat_ratio = fat_kcal / (cals + 1)
            return 0.35 - fat_ratio  # >= 0

        def carb_max_constraint(x):
            h_local = self.horizon
            cals = x[:h_local]
            carbs = x[2*h_local:3*h_local]
            carbs_ratio = carbs * 4 / (cals + 1)
            return 0.50 - carbs_ratio  # >= 0

        constraints = [
            {'type': 'ineq', 'fun': fat_min_constraint},
            {'type': 'ineq', 'fun': fat_max_constraint},
            {'type': 'ineq', 'fun': carb_max_constraint},
        ]

        # 求解
        result = minimize(
            self.objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-6}
        )

        if not result.success:
            print(f"MPC 优化警告: {result.message}")

        # 解析结果
        cals = result.x[:h]
        proteins = result.x[h:2*h]
        carbs = result.x[2*h:3*h]
        fats_kcal = cals - proteins * 4 - carbs * 4
        fats = np.maximum(fats_kcal / 9, 0)

        daily_plan = []
        weight = self.current_weight
        predicted_weights = []

        for t in range(h):
            daily_plan.append({
                'day': t + 1,
                'total_cal': round(float(cals[t]), 0),
                'protein_g': round(float(proteins[t]), 0),
                'carbs_g': round(float(carbs[t]), 0),
                'fat_g': round(float(fats[t]), 0),
                'protein_ratio': round(float(proteins[t] * 4 / cals[t]) * 100, 1),
                'carbs_ratio': round(float(carbs[t] * 4 / cals[t]) * 100, 1),
                'fat_ratio': round(float(fats[t] * 9 / cals[t]) * 100, 1),
            })

            tdee_t = self._compute_tdee_at_weight(weight)
            deficit = tdee_t - cals[t]  # >0 = 消耗>摄入 = 减重
            weight -= deficit / 7700
            predicted_weights.append(round(weight, 2))

        total_deficit = sum(self._compute_tdee_at_weight(
            self.current_weight - self.daily_deficit_target * t / 7700
        ) - cals[t] for t in range(h))

        return {
            'daily_plan': daily_plan,
            'predicted_weights': predicted_weights,
            'total_deficit': round(total_deficit, 0),
            'target_deficit': round(self.daily_deficit_target * h, 0),
            'current_weight': round(self.current_weight, 1),
            'tdee': round(self.tdee, 0),
            'bmr': round(self.bmr, 0),
            'target_cal': round(target_cal, 0),
            'loss': round(float(result.fun), 4),
            'converged': result.success,
        }


def get_today_plan() -> Dict:
    """获取今天的第一天营养目标"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.db_layer import get_recent_logs

    logs = get_recent_logs(14)
    current_weight = logs[-1]['weight'] if logs else 75.0

    mpc = MPCOptimizer(
        horizon=7,
        current_weight=current_weight,
        height_cm=175,
        age=28,
        sex='male',
        activity_level=1.2,
        target_weight_loss_weekly=0.4
    )

    result = mpc.optimize()
    return result


if __name__ == "__main__":
    result = get_today_plan()

    print("=" * 60)
    print("MPC 优化结果 - 7天营养规划")
    print("=" * 60)
    print(f"当前体重: {result['current_weight']} kg")
    print(f"BMR: {result['bmr']} kcal | TDEE: {result['tdee']} kcal")
    print(f"目标热量: ~{result['target_cal']} kcal/天")
    print(f"优化收敛: {result['converged']}")
    print("-" * 60)

    for day in result['daily_plan']:
        print(f"\nDay {day['day']}: "
              f"{day['total_cal']:.0f} kcal | "
              f"P:{day['protein_g']:.0f}g({day['protein_ratio']}%) "
              f"C:{day['carbs_g']:.0f}g({day['carbs_ratio']}%) "
              f"F:{day['fat_g']:.0f}g({day['fat_ratio']}%)")

    print(f"\n预测体重走势: {result['predicted_weights']}")
    print(f"总热量缺口: {result['total_deficit']} kcal (目标: {result['target_deficit']} kcal)")
