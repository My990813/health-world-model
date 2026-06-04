"""
导出 JEPA 模型权重为 JSON，供前端 JavaScript 加载。
模型结构（来自训练日志）：
  encoder.encoder (MLP):  (112 -> 128 -> 64 -> 32)  3层
  day_encoder (MLP):     (8   -> 64  -> 32)          2层
  predictor (MLP):       (40  -> 96  -> 64 -> 32)   3层
  decoder (MLP):         (32  -> 48  -> 16 -> 2)    3层
"""
import pickle, json, numpy as np, os, sys

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, BASE)

from models.jepa_model import relu

MODEL_FILE = os.path.join(BASE, 'data', 'jepa_model.pkl')
STATS_FILE = os.path.join(BASE, 'data', 'jepa_stats.json')
OUT_FILE   = os.path.join(BASE, 'frontend', 'jepa_weights.json')

# ---- 工具 ----

def mlp_to_dict(mlp, name):
    """把 MLP 对象转成 JSON 友好的 dict"""
    layers = []
    for i, layer in enumerate(mlp.layers):
        W = layer['W']  # ndarray
        b = layer['b']
        layers.append({
            'W': W.tolist(),
            'b': b.tolist(),
            'shape_W': list(W.shape),
            'shape_b': list(b.shape),
        })
    return {'name': name, 'n_layers': mlp.n_layers, 'layers': layers}


def export():
    print("加载模型:", MODEL_FILE)
    with open(MODEL_FILE, 'rb') as f:
        data = pickle.load(f)

    model = data['model']
    stats = data.get('stats', {})

    # ---- 组装导出结构 ----
    export_data = {
        'version': 'jepa-v1',
        'input_dim': int(model.input_dim),
        'latent_dim': int(model.latent_dim),
        'context_len': int(model.context_len),
        'stats': stats,
        'networks': {
            'encoder': mlp_to_dict(model.encoder.encoder, 'encoder'),
            'day_encoder': mlp_to_dict(model.day_encoder, 'day_encoder'),
            'predictor': mlp_to_dict(model.predictor, 'predictor'),
            'decoder': mlp_to_dict(model.decoder, 'decoder'),
        }
    }

    # ---- 写 JSON ----
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, 'w') as f:
        json.dump(export_data, f, indent=2)

    print("导出成功:", OUT_FILE)
    total = 0
    for net_name, net in export_data['networks'].items():
        n = sum(len(l['W']) * len(l['W'][0]) + len(l['b']) for l in net['layers'])
        total += n
        print(f"  {net_name}: {net['n_layers']} 层, {n} 参数")
    print(f"  合计: {total} 参数")
    print(f"  文件大小: {os.path.getsize(OUT_FILE)//1024} KB")


if __name__ == '__main__':
    export()
