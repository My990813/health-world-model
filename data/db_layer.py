"""
数据存储层：health_log 数据库管理
- SQLite 数据库存储每日健康时序数据
- 模拟历史数据生成（用于初始 JEPA 训练）
- 食材营养数据库
"""

import sqlite3
import json
import os
import random
import math
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "health_log.db")


def get_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """初始化数据库表结构"""
    conn = get_connection()
    cursor = conn.cursor()

    # 每日健康记录表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL UNIQUE,
            weight REAL NOT NULL,
            breakfast_cal REAL DEFAULT 0,
            lunch_cal REAL DEFAULT 0,
            dinner_cal REAL DEFAULT 0,
            snack_cal REAL DEFAULT 0,
            total_cal REAL GENERATED ALWAYS AS (
                breakfast_cal + lunch_cal + dinner_cal + snack_cal
            ) STORED,
            protein_g REAL DEFAULT 0,
            carbs_g REAL DEFAULT 0,
            fat_g REAL DEFAULT 0,
            exercise_min REAL DEFAULT 0,
            exercise_type TEXT DEFAULT 'rest',
            sleep_hours REAL DEFAULT 7,
            meal_rating REAL DEFAULT 3,
            water_ml REAL DEFAULT 2000,
            notes TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # 食材营养数据库
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS food_database (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            cal_per_100g REAL NOT NULL,
            protein_per_100g REAL NOT NULL,
            carbs_per_100g REAL NOT NULL,
            fat_per_100g REAL NOT NULL,
            fiber_per_100g REAL DEFAULT 0,
            serving_g REAL DEFAULT 100,
            is_staple INTEGER DEFAULT 0,
            is_protein_source INTEGER DEFAULT 0,
            is_vegetable INTEGER DEFAULT 0,
            is_fruit INTEGER DEFAULT 0,
            available INTEGER DEFAULT 1
        )
    """)

    # 模型训练快照表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            version TEXT NOT NULL,
            train_date TEXT NOT NULL,
            train_loss REAL DEFAULT 0,
            val_loss REAL DEFAULT 0,
            config_json TEXT DEFAULT '{}',
            weights_path TEXT DEFAULT '',
            notes TEXT DEFAULT ''
        )
    """)

    # MPC 优化结果表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mpc_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_date TEXT NOT NULL,
            horizon_days INTEGER DEFAULT 7,
            target_cal_json TEXT DEFAULT '[]',
            target_protein_json TEXT DEFAULT '[]',
            target_carbs_json TEXT DEFAULT '[]',
            predicted_weight_json TEXT DEFAULT '[]',
            jepa_prediction_json TEXT DEFAULT '{}',
            executed TEXT DEFAULT '',
            actual_cal REAL DEFAULT 0,
            actual_weight REAL DEFAULT 0,
            notes TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()


def generate_synthetic_history(days=120):
    """
    生成模拟历史数据用于 JEPA 初始训练
    模拟一个 75kg 男性，目标减到 65kg，12周周期
    """
    conn = get_connection()

    # 参数
    base_weight = 78.0
    target_weight = 68.0
    bmr = 1700  # 基础代谢
    tef_factor = 0.1  # 食物热效应
    activity_factor_range = (0.15, 0.35)  # 运动消耗比例

    foods = load_food_database()

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    cursor = conn.cursor()

    for i in range(days):
        current_date = start_date + timedelta(days=i)

        # 检查是否已有数据
        existing = cursor.execute(
            "SELECT id FROM daily_log WHERE date = ?", (current_date.isoformat(),)
        ).fetchone()
        if existing:
            continue

        # 模拟减重曲线（非线性的，前期快后期慢）
        progress = i / days
        weight_loss = (base_weight - target_weight) * (
            1 - math.exp(-3 * progress)  # 指数衰减式减重
        )
        daily_weight = base_weight - weight_loss

        # 添加随机波动（±0.3kg）
        weight_noise = random.gauss(0, 0.15)
        daily_weight += weight_noise

        # 模拟饮食摄入（随减重进程逐渐下降热量）
        target_cal = bmr * 0.8 + random.randint(-150, 150)  # 20%热量缺口
        actual_cal = target_cal + random.randint(-200, 200)
        actual_cal = max(1200, actual_cal)  # 不低于1200

        # 分配到三餐
        b_ratio = random.uniform(0.25, 0.35)
        l_ratio = random.uniform(0.30, 0.40)
        d_ratio = 1 - b_ratio - l_ratio - random.uniform(0.05, 0.15)
        d_ratio = max(0.20, d_ratio)

        breakfast_cal = int(actual_cal * b_ratio)
        lunch_cal = int(actual_cal * (1 - b_ratio) * l_ratio / (l_ratio + 1 - b_ratio - l_ratio))
        dinner_cal = actual_cal - breakfast_cal - lunch_cal - random.randint(50, 200)
        snack_cal = actual_cal - breakfast_cal - lunch_cal - dinner_cal

        # 蛋白质（高蛋白策略）
        protein_per_kg = random.uniform(1.5, 2.2)
        protein_g = daily_weight * protein_per_kg + random.uniform(-5, 5)

        # 碳水和脂肪（剩余热量分配）
        remaining_cal = actual_cal - protein_g * 4 - random.uniform(-50, 50)
        fat_ratio = random.uniform(0.25, 0.35)
        fat_g = (remaining_cal * fat_ratio) / 9
        carbs_g = (remaining_cal * (1 - fat_ratio)) / 4

        # 运动
        if random.random() < 0.7:  # 70%概率运动
            exercise_min = random.choice([20, 30, 40, 45, 50, 60, 75])
            exercise_type = random.choice(['running', 'walking', 'cycling', 'swimming', 'strength', 'hiit', 'yoga'])
        else:
            exercise_min = 0
            exercise_type = 'rest'

        # 睡眠
        sleep_hours = round(random.uniform(6.0, 8.5), 1)

        # 餐评
        meal_rating = round(random.uniform(2.0, 5.0), 1)

        cursor.execute("""
            INSERT OR IGNORE INTO daily_log (
                date, weight, breakfast_cal, lunch_cal, dinner_cal, snack_cal,
                protein_g, carbs_g, fat_g, exercise_min, exercise_type,
                sleep_hours, meal_rating, water_ml
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            current_date.isoformat(), round(daily_weight, 1),
            breakfast_cal, lunch_cal, max(0, dinner_cal), max(0, snack_cal),
            round(protein_g, 1), round(max(0, carbs_g), 1), round(max(0, fat_g), 1),
            exercise_min, exercise_type, sleep_hours, meal_rating,
            random.randint(1500, 3000)
        ))

    conn.commit()
    conn.close()
    print(f"已生成 {days} 天模拟历史数据 ({start_date} ~ {end_date})")


def load_food_database():
    """加载食材营养数据库"""
    foods = [
        # 主食类
        ("白米饭", "staple", 116, 2.6, 25.9, 0.3, 0.3, 200, 1, 0, 0, 0),
        ("糙米饭", "staple", 123, 2.7, 25.5, 1.0, 1.8, 200, 1, 0, 0, 0),
        ("全麦面包", "staple", 247, 13.0, 41.0, 3.4, 6.0, 60, 1, 0, 0, 0),
        ("燕麦片", "staple", 379, 13.2, 67.7, 6.5, 10.6, 40, 1, 0, 0, 0),
        ("红薯", "staple", 86, 1.6, 20.1, 0.1, 3.0, 200, 1, 0, 0, 0),
        ("玉米", "staple", 112, 4.1, 22.8, 1.2, 2.9, 200, 1, 0, 0, 0),
        ("意大利面(熟)", "staple", 157, 5.8, 30.6, 0.9, 1.8, 200, 1, 0, 0, 0),

        # 高蛋白
        ("鸡胸肉", "protein", 133, 31.0, 0, 3.6, 0, 150, 0, 1, 0, 0),
        ("三文鱼", "protein", 139, 21.6, 0, 5.9, 0, 150, 0, 1, 0, 0),
        ("牛肉(瘦)", "protein", 125, 26.0, 0, 2.5, 0, 150, 0, 1, 0, 0),
        ("虾仁", "protein", 87, 18.6, 1.0, 0.8, 0, 120, 0, 1, 0, 0),
        ("鸡蛋(全)", "protein", 144, 13.3, 1.5, 9.5, 0, 60, 0, 1, 0, 0),
        ("蛋白", "protein", 48, 11.0, 0.7, 0.2, 0, 33, 0, 1, 0, 0),
        ("豆腐", "protein", 73, 8.1, 2.8, 3.7, 0.4, 150, 0, 1, 0, 0),
        ("希腊酸奶", "protein", 59, 10.0, 3.6, 0.7, 0, 150, 0, 1, 0, 0),
        ("牛奶", "protein", 42, 3.4, 5.0, 1.0, 0, 250, 0, 1, 0, 0),

        # 蔬菜
        ("西蓝花", "vegetable", 34, 4.1, 6.6, 0.4, 2.6, 150, 0, 0, 1, 0),
        ("菠菜", "vegetable", 23, 2.9, 3.6, 0.4, 2.2, 100, 0, 0, 1, 0),
        ("番茄", "vegetable", 19, 0.9, 4.0, 0.2, 1.2, 150, 0, 0, 1, 0),
        ("黄瓜", "vegetable", 15, 0.8, 3.1, 0.1, 0.5, 200, 0, 0, 1, 0),
        ("生菜", "vegetable", 13, 1.3, 2.1, 0.3, 0.7, 80, 0, 0, 1, 0),
        ("胡萝卜", "vegetable", 37, 0.9, 8.1, 0.2, 2.8, 100, 0, 0, 1, 0),
        ("蘑菇", "vegetable", 22, 3.1, 3.3, 0.3, 1.0, 100, 0, 0, 1, 0),
        ("彩椒", "vegetable", 26, 1.0, 6.3, 0.2, 1.7, 120, 0, 0, 1, 0),

        # 水果
        ("苹果", "fruit", 53, 0.3, 14.0, 0.2, 2.4, 200, 0, 0, 0, 1),
        ("香蕉", "fruit", 89, 1.1, 22.8, 0.3, 2.6, 120, 0, 0, 0, 1),
        ("蓝莓", "fruit", 57, 0.7, 14.5, 0.3, 2.4, 100, 0, 0, 0, 1),
        ("橙子", "fruit", 48, 0.9, 11.8, 0.1, 2.4, 200, 0, 0, 0, 1),
        ("葡萄", "fruit", 69, 0.7, 18.1, 0.2, 0.9, 150, 0, 0, 0, 1),

        # 健康脂肪
        ("牛油果", "fat", 160, 2.0, 8.5, 14.7, 6.7, 100, 0, 0, 0, 0),
        ("坚果混合", "fat", 607, 20.0, 21.0, 54.0, 7.0, 30, 0, 0, 0, 0),
        ("橄榄油", "fat", 884, 0, 0, 100, 0, 10, 0, 0, 0, 0),
    ]

    conn = get_connection()
    cursor = conn.cursor()

    for food in foods:
        cursor.execute("""
            INSERT OR IGNORE INTO food_database (
                name, category, cal_per_100g, protein_per_100g,
                carbs_per_100g, fat_per_100g, fiber_per_100g,
                serving_g, is_staple, is_protein_source, is_vegetable, is_fruit
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, food)

    conn.commit()
    conn.close()
    return foods


def get_recent_logs(days=14):
    """获取最近N天的健康记录"""
    conn = get_connection()
    cursor = conn.cursor()
    rows = cursor.execute("""
        SELECT * FROM daily_log
        WHERE date <= date('now')
        ORDER BY date DESC
        LIMIT ?
    """, (days,)).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]  # 按日期正序


def get_all_logs():
    """获取所有健康记录"""
    conn = get_connection()
    cursor = conn.cursor()
    rows = cursor.execute("""
        SELECT * FROM daily_log ORDER BY date ASC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_daily_log(entry: dict):
    """添加每日记录"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO daily_log (
                date, weight, breakfast_cal, lunch_cal, dinner_cal, snack_cal,
                protein_g, carbs_g, fat_g, exercise_min, exercise_type,
                sleep_hours, meal_rating, water_ml, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.get('date'),
            entry.get('weight'),
            entry.get('breakfast_cal', 0),
            entry.get('lunch_cal', 0),
            entry.get('dinner_cal', 0),
            entry.get('snack_cal', 0),
            entry.get('protein_g', 0),
            entry.get('carbs_g', 0),
            entry.get('fat_g', 0),
            entry.get('exercise_min', 0),
            entry.get('exercise_type', 'rest'),
            entry.get('sleep_hours', 7),
            entry.get('meal_rating', 3),
            entry.get('water_ml', 2000),
            entry.get('notes', '')
        ))
        conn.commit()
        print(f"记录已入库: {entry.get('date')} 体重={entry.get('weight')}kg")
    except sqlite3.IntegrityError:
        print(f"日期 {entry.get('date')} 已有记录，跳过")
    finally:
        conn.close()


def get_foods_by_category(category=None):
    """获取食材列表"""
    conn = get_connection()
    cursor = conn.cursor()
    if category:
        rows = cursor.execute(
            "SELECT * FROM food_database WHERE category = ? AND available = 1",
            (category,)
        ).fetchall()
    else:
        rows = cursor.execute(
            "SELECT * FROM food_database WHERE available = 1"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    load_food_database()
    generate_synthetic_history(120)
    print("数据存储层初始化完成!")
