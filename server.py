"""
JEPA Health Model - Flask API Server
五层架构的统一接口
"""

import os
import sys
import json
import numpy as np
from datetime import datetime, date

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db_layer import (
    init_db, load_food_database, generate_synthetic_history,
    get_recent_logs, get_all_logs, add_daily_log, get_foods_by_category
)
from optimizer.mpc_optimizer import MPCOptimizer
from optimizer.mc_meals import mc_generate_meals, generate_meal_for_today
from models.jepa_model import (
    load_jepa_model, predict_weight_trend, simple_physics_predict
)

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# 全局缓存
MODEL_CACHE = {'model': None, 'stats': None, 'loaded': False}


def _ensure_model():
    """懒加载 JEPA 模型"""
    if not MODEL_CACHE['loaded']:
        # 确保类在当前命名空间中
        from models.jepa_model import JEPAWorldModel
        MODEL_CACHE['model'], MODEL_CACHE['stats'] = load_jepa_model()
        MODEL_CACHE['loaded'] = True
    return MODEL_CACHE['model']


# ==================== 页面路由 ====================

@app.route('/')
def index():
    return send_file(os.path.join(app.static_folder, 'index.html'))


# ==================== 数据接口 ====================

@app.route('/api/init', methods=['POST'])
def init_system():
    """初始化系统（数据库+食材+模拟数据+训练模型）"""
    init_db()
    load_food_database()
    generate_synthetic_history(120)
    return jsonify({'status': 'ok', 'message': '系统初始化完成'})


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """获取健康记录"""
    days = request.args.get('days', 120, type=int)
    logs = get_recent_logs(days)
    return jsonify(logs)


@app.route('/api/logs/all', methods=['GET'])
def get_all():
    """获取所有记录"""
    logs = get_all_logs()
    return jsonify(logs)


@app.route('/api/logs/add', methods=['POST'])
def add_log():
    """添加每日记录"""
    entry = request.json
    add_daily_log(entry)
    return jsonify({'status': 'ok'})


# ==================== JEPA 世界模型接口 ====================

@app.route('/api/jepa/simulate', methods=['POST'])
def simulate():
    """虚拟试餐模拟器"""
    data = request.json
    diet_plan = data.get('diet_plan', [])
    horizon = data.get('horizon', 7)

    logs = get_recent_logs(14)
    model = _ensure_model()

    result = predict_weight_trend(logs, diet_plan, model)

    # 附加JEPA模型信息
    result['model_loaded'] = model is not None
    result['method'] = 'jepa' if model else 'simple_physics'

    return jsonify(result)


@app.route('/api/jepa/status', methods=['GET'])
def jepa_status():
    """检查 JEPA 模型状态"""
    model = _ensure_model()
    stats = MODEL_CACHE['stats']
    return jsonify({
        'loaded': model is not None,
        'stats': stats,
        'has_enough_data': len(get_recent_logs(14)) >= 14,
    })


# ==================== MPC 优化接口 ====================

@app.route('/api/mpc/optimize', methods=['POST'])
def mpc_optimize():
    """MPC 滚动优化"""
    data = request.get_json(force=True, silent=True) or {}
    weight = data.get('current_weight')
    horizon = data.get('horizon', 7)
    target_loss = data.get('target_loss_weekly', 0.4)
    activity = data.get('activity_level', 1.2)

    if weight is None:
        logs = get_recent_logs(1)
        weight = logs[-1]['weight'] if logs else 75.0

    mpc = MPCOptimizer(
        horizon=horizon,
        current_weight=float(weight),
        target_weight_loss_weekly=float(target_loss),
        activity_level=float(activity),
    )

    result = mpc.optimize()
    return jsonify(result)


# ==================== MC 配餐接口 ====================

@app.route('/api/meals/generate', methods=['POST'])
def generate_meals():
    """MC 蒙特卡洛配餐"""
    data = request.json or {}
    target = data.get('target', {})
    n_samples = data.get('n_samples', 3000)
    top_k = data.get('top_k', 3)

    plans = mc_generate_meals(
        target_cal=target.get('total_cal', 1500),
        target_protein=target.get('protein_g', 120),
        target_carbs=target.get('carbs_g', 150),
        target_fat=target.get('fat_g', 50),
        n_samples=n_samples,
        top_k=top_k,
    )

    return jsonify(plans)


# ==================== 食材接口 ====================

@app.route('/api/foods', methods=['GET'])
def get_foods():
    """获取食材列表"""
    category = request.args.get('category')
    foods = get_foods_by_category(category)
    return jsonify(foods)


# ==================== 综合面板数据 ====================

@app.route('/api/dashboard', methods=['GET'])
def dashboard():
    """仪表盘综合数据"""
    logs = get_recent_logs(14)
    all_logs = get_all_logs()

    model = _ensure_model()

    return jsonify({
        'recent_logs': logs,
        'total_days': len(all_logs),
        'latest_weight': logs[-1]['weight'] if logs else None,
        'model_loaded': model is not None,
        'has_enough_data': len(logs) >= 14,
    })


# ==================== 启动 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("JEPA Health Model Server")
    print("=" * 60)

    # 确保数据库初始化
    init_db()
    load_food_database()

    # 检查是否有数据
    logs = get_recent_logs(1)
    if not logs:
        print("数据库为空，生成模拟数据...")
        generate_synthetic_history(120)

    # 尝试加载模型
    model = _ensure_model()
    if model:
        print("JEPA 模型已加载")
    else:
        print("JEPA 模型未找到（将使用物理模型 fallback）")

    print("服务启动: http://localhost:5001")
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)
