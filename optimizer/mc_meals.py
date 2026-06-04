"""
MC 蒙特卡洛配餐层 (Layer 4)
============================
基于 MPC 输出的当日营养目标，用蒙特卡洛采样从食材库生成三餐方案

约束：
  - 每餐总量贴近目标
  - 单餐某食材不能超过该餐 40% 热量
  - 三餐热量分配：早 30% / 午 40% / 晚 25% / 加餐 5%
  - 营养配比符合 MPC 目标
"""

import numpy as np
import random
import os
import json
from typing import Dict, List, Tuple

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "health_log.db")


def _get_foods():
    """从数据库获取食材"""
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    rows = cursor.execute(
        "SELECT * FROM food_database WHERE available = 1"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _food_cal(food: Dict, serving_g: float) -> float:
    """计算食材在给定份量下的热量"""
    return food['cal_per_100g'] * serving_g / 100


def _food_macros(food: Dict, serving_g: float) -> Dict:
    """计算食材在给定份量下的宏量营养素"""
    factor = serving_g / 100
    return {
        'cal': food['cal_per_100g'] * factor,
        'protein': food['protein_per_100g'] * factor,
        'carbs': food['carbs_per_100g'] * factor,
        'fat': food['fat_per_100g'] * factor,
    }


def mc_generate_meals(
    target_cal: float,
    target_protein: float,
    target_carbs: float,
    target_fat: float,
    n_samples: int = 5000,
    top_k: int = 5,
    meal_split: Dict = None
) -> List[Dict]:
    """
    蒙特卡洛配餐

    Args:
        target_cal: 目标总热量
        target_protein: 目标总蛋白(g)
        target_carbs: 目标总碳水(g)
        target_fat: 目标总脂肪(g)
        n_samples: 采样次数
        top_k: 返回前K个最佳方案
        meal_split: 三餐热量分配比例

    Returns:
        top_k 个三餐方案，按与目标的贴近度排序
    """
    if meal_split is None:
        meal_split = {
            'breakfast': {'ratio': 0.30, 'name': '早餐'},
            'lunch': {'ratio': 0.40, 'name': '午餐'},
            'dinner': {'ratio': 0.25, 'name': '晚餐'},
            'snack': {'ratio': 0.05, 'name': '加餐'},
        }

    foods = _get_foods()
    foods_by_cat = {}
    for f in foods:
        cat = f['category']
        if cat not in foods_by_cat:
            foods_by_cat[cat] = []
        foods_by_cat[cat].append(f)

    best_plans = []
    max_single_food_ratio = 0.40  # 单食材不超过40%热量

    for _ in range(n_samples):
        plan = _sample_one_plan(
            foods, foods_by_cat, target_cal, target_protein,
            target_carbs, target_fat, meal_split, max_single_food_ratio
        )
        if plan:
            best_plans.append(plan)

    # 排序：按综合得分（越小越好）
    best_plans.sort(key=lambda p: p['score'])
    return best_plans[:top_k]


def _sample_one_plan(
    foods: List[Dict],
    foods_by_cat: Dict,
    target_cal: float,
    target_protein: float,
    target_carbs: float,
    target_fat: float,
    meal_split: Dict,
    max_ratio: float
) -> Dict:
    """采样一个完整的三餐方案"""
    total_cal = 0
    total_protein = 0
    total_carbs = 0
    total_fat = 0
    meals = []

    for meal_key, meal_info in meal_split.items():
        meal_target_cal = target_cal * meal_info['ratio']
        meal_items = _sample_meal(
            foods, foods_by_cat, meal_target_cal,
            max_ratio, meal_key
        )

        if not meal_items:
            return None

        meal_cal = sum(i['macros']['cal'] for i in meal_items)
        meal_protein = sum(i['macros']['protein'] for i in meal_items)
        meal_carbs = sum(i['macros']['carbs'] for i in meal_items)
        meal_fat = sum(i['macros']['fat'] for i in meal_items)

        total_cal += meal_cal
        total_protein += meal_protein
        total_carbs += meal_carbs
        total_fat += meal_fat

        meals.append({
            'meal': meal_info['name'],
            'target_cal': round(meal_target_cal, 0),
            'actual_cal': round(meal_cal, 0),
            'items': meal_items,
        })

    # 计算综合得分
    cal_err = abs(total_cal - target_cal) / target_cal
    protein_err = abs(total_protein - target_protein) / (target_protein + 1)
    carbs_err = abs(total_carbs - target_carbs) / (target_carbs + 1)
    fat_err = abs(total_fat - target_fat) / (target_fat + 1)

    score = (cal_err * 3 + protein_err * 2 + carbs_err + fat_err)

    return {
        'score': round(score, 4),
        'total_cal': round(total_cal, 0),
        'total_protein': round(total_protein, 1),
        'total_carbs': round(total_carbs, 1),
        'total_fat': round(total_fat, 1),
        'p_ratio': round(total_protein * 4 / total_cal * 100, 1) if total_cal > 0 else 0,
        'c_ratio': round(total_carbs * 4 / total_cal * 100, 1) if total_cal > 0 else 0,
        'f_ratio': round(total_fat * 9 / total_cal * 100, 1) if total_cal > 0 else 0,
        'meals': meals,
    }


def _sample_meal(
    foods: List[Dict],
    foods_by_cat: Dict,
    target_cal: float,
    max_ratio: float,
    meal_type: str
) -> List[Dict]:
    """采样一餐的食材组合"""
    n_items = random.randint(3, 6)  # 每餐3~6种食材

    # 根据餐次选择食材类别偏好
    if meal_type == 'breakfast':
        preferred_cats = ['staple', 'protein', 'fruit']
    elif meal_type == 'lunch':
        preferred_cats = ['staple', 'protein', 'vegetable']
    elif meal_type == 'dinner':
        preferred_cats = ['protein', 'vegetable', 'staple']
    else:  # snack
        preferred_cats = ['fruit', 'protein']
        n_items = random.randint(1, 3)

    items = []
    remaining_cal = target_cal

    for _ in range(n_items):
        if remaining_cal <= 20:
            break

        # 选择类别
        cat = random.choice(preferred_cats)
        if cat not in foods_by_cat or not foods_by_cat[cat]:
            continue

        food = random.choice(foods_by_cat[cat])

        # 随机份量 (50%~150% of serving)
        base_serving = food['serving_g']
        serving = base_serving * random.uniform(0.5, 1.5)
        serving = max(10, serving)

        macros = _food_macros(food, serving)
        cal = macros['cal']

        # 检查不超过单餐 40%
        if cal > target_cal * max_ratio:
            serving = target_cal * max_ratio * 100 / (food['cal_per_100g'] + 1) * 0.9
            macros = _food_macros(food, serving)
            cal = macros['cal']

        if cal > remaining_cal:
            continue

        items.append({
            'name': food['name'],
            'category': cat,
            'serving_g': round(serving, 0),
            'macros': {k: round(v, 1) for k, v in macros.items()},
        })

        remaining_cal -= cal

    return items if items else None


def generate_meal_for_today(target_plan: Dict) -> Dict:
    """
    为今天生成配餐方案（使用MPC输出的第一天目标）

    Args:
        target_plan: MPC 优化结果中的 daily_plan[0]
    """
    day = target_plan['daily_plan'][0]  # 第一天的营养目标

    plans = mc_generate_meals(
        target_cal=day['total_cal'],
        target_protein=day['protein_g'],
        target_carbs=day['carbs_g'],
        target_fat=day['fat_g'],
        n_samples=3000,
        top_k=3,
    )

    if not plans:
        return {'error': '未找到满足约束的配餐方案'}

    return {
        'best_plan': plans[0],
        'alternatives': plans[1:],
        'target': day,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 模拟 MPC 输出的第一天目标
    target = {
        'total_cal': 1530,
        'protein_g': 127,
        'carbs_g': 151,
        'fat_g': 46,
    }

    print("=" * 60)
    print("MC 蒙特卡洛配餐")
    print(f"目标: {target['total_cal']}kcal | P:{target['protein_g']}g | C:{target['carbs_g']}g | F:{target['fat_g']}g")
    print("=" * 60)

    result = generate_meal_for_today({'daily_plan': [target]})

    if 'error' in result:
        print(result['error'])
    else:
        best = result['best_plan']
        print(f"\n最佳方案 (得分: {best['score']:.4f})")
        print(f"总热量: {best['total_cal']} kcal | P:{best['total_protein']}g({best['p_ratio']}%) "
              f"C:{best['total_carbs']}g({best['c_ratio']}%) F:{best['total_fat']}g({best['f_ratio']}%)")

        for meal in best['meals']:
            print(f"\n--- {meal['meal']} (目标:{meal['target_cal']:.0f}kcal 实际:{meal['actual_cal']:.0f}kcal) ---")
            for item in meal['items']:
                print(f"  {item['name']:10s} {item['serving_g']:.0f}g | "
                      f"{item['macros']['cal']:.0f}kcal "
                      f"P:{item['macros']['protein']:.1f}g "
                      f"C:{item['macros']['carbs']:.1f}g "
                      f"F:{item['macros']['fat']:.1f}g")
