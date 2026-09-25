# JEPA Health World Model · 健康世界模型

Use this self-made website: <https://my990813.github.io/health-world-model/>

基于 **JEPA（Joint-Embedding Predictive Architecture）世界模型 + MPC 滚动优化 + LP/MC 配餐** 的个人健康管理系统。

输入你的身体数据与饮食记录，系统学习 `Body(t+1) = F(Diet(t), Body(t))` 的人体动力学，给出未来 7 天的营养规划，并直接生成可直接照做的三餐方案。

线上版为**纯前端实现**（模型权重嵌入 HTML，浏览器本地计算，数据存 localStorage），无需后端即可使用。

---

## 一、五层架构

```
┌─────────────────────────────────────────────────────┐
│  Layer 5  前端交互 (frontend/index.html)             │
│           SPA · Chart.js 可视化 · localStorage       │
├─────────────────────────────────────────────────────┤
│  Layer 4  配餐求解                                    │
│           LP 线性规划 (tableau simplex + Big-M)       │
│           MC 蒙特卡洛采样 (fallback)                   │
├─────────────────────────────────────────────────────┤
│  Layer 3  MPC 滚动优化 (optimizer/mpc_optimizer.py)  │
│           scipy SLSQP · 7 天营养规划                  │
├─────────────────────────────────────────────────────┤
│  Layer 2  JEPA 世界模型 (models/jepa_model.py)       │
│           纯 NumPy MLP · 随机 mask 预训练 + 监督微调   │
├─────────────────────────────────────────────────────┤
│  Layer 1  数据存储 (data/db_layer.py)                │
│           SQLite · 每日健康时序 · 食材营养库           │
└─────────────────────────────────────────────────────┘
```

## 二、项目流程（端到端）

### Step 1 · 数据采集与初始化

- `data/db_layer.py` 初始化 SQLite（`data/health_log.db`），建两张表：
  - `daily_log`：每日体重、四餐热量、蛋白/碳水/脂肪、运动
  - `food_database`：食材营养库（热量、蛋白、碳水、脂肪、GI、纤维、维C，每 100g）
- 首次运行调用 `generate_synthetic_history(120)` 生成 120 天模拟历史，解决"新用户无数据无法训练"的冷启动问题
- 之后每日真实记录持续写入，模型定期用真实数据重训

### Step 2 · JEPA 世界模型训练

`models/jepa_model.py` —— 纯 NumPy 实现的 MLP + 完整反向传播（CPU 友好，无 PyTorch 依赖）。

模型结构：

| 模块 | 结构 | 作用 |
|------|------|------|
| encoder | 112 → 128 → 64 → 32 | 将 Diet+Body 状态编码到隐空间 |
| day_encoder | 8 → 64 → 32 | 编码星期几等日历特征 |
| predictor | 40 → 96 → 64 → 32 | 在**隐空间**预测 t+1 状态 |
| decoder | 32 → 48 → 16 → 2 | 从隐向量解码出体重等物理量 |

两阶段训练：

1. **Phase 1 自监督预训练**：随机 mask 若干天的输入，用 predictor 在隐空间补全，损失为 **Cosine Embedding Loss**（对齐方向而非数值，防止表征坍缩）
2. **Phase 2 监督微调**：用体重真实值微调 decoder（MSE Loss）

学到的映射：`Body(t+1) = F(Diet(t), Body(t))`，即"今天怎么吃 → 明天身体怎么变"。

训练产物：`data/jepa_model.pkl`（权重）+ `data/jepa_stats.json`（归一化参数）。

### Step 3 · MPC 滚动优化（7 天规划）

`optimizer/mpc_optimizer.py` 用 JEPA 模型作为状态转移函数，做 Model Predictive Control：

- **优化变量**：未来 7 天每天的目标热量 / 蛋白 / 碳水
- **优化目标**：周度减重速率贴近 0.3~0.5 kg/周（可配置）
- **硬约束**：
  - 热量 ∈ [1000, TDEE]
  - 蛋白 ≥ 1.6 g/kg 体重（减脂保肌肉）
  - 脂肪供能比 20%~35%，碳水 30%~45%
- **求解器**：`scipy.optimize.minimize`（SLSQP），滚动时域——每天用最新体重重解一次

输出：每日营养目标（如"1750 kcal / 蛋白 120g"），交给 Step 4。

### Step 4 · 配餐求解（LP 主 + MC 兜底）

**方案 A · LP 线性规划（主路径，~50ms）**

前端 `frontend/index.html` 内置纯 JS 实现的 tableau 单纯形法（Big-M）：

- **决策变量**：~200 个（每种候选食物的克数 + 营养偏差松弛变量）
- **约束**：~220 条（热量/蛋白/碳水/脂肪等式、份量上限、餐次热量上下限）
- **目标**：最小化营养偏差
- 高 GI 食物（GI>70）份量上限减半，中 GI（55~70）乘 0.7，控血糖
- 通过 cost perturbation 一次求解生成 3 个不同方案，避免"每天吃一样的"

**方案 B · MC 蒙特卡洛（fallback，500 次采样）**

`optimizer/mc_meals.py`：随机采样食物组合与份量，保留营养最贴合目标的方案。LP 失败时快速兜底，保证任何情况下都有输出。

公共约束：三餐热量分配 早 30% / 午 40% / 晚 25% / 加餐 5%，单食材 ≤ 该餐热量 40%。

### Step 5 · 前端呈现与闭环

`frontend/index.html`（单文件 SPA）：

- 体重趋势、营养摄入图表（Chart.js）
- 输入身体数据 → 规划 → 配餐 → 展示三餐方案（食材克数、营养明细、健康提示）
- 用户按方案执行 → 记录回写 → 下一轮 MPC 用新数据滚动优化，形成**闭环**

### 前端独立运行说明

训练好的 JEPA 权重由 `models/export_weights.py` 导出为 JSON，再经 `models/embed_weights.py` 直接嵌入 HTML `<script>` 标签（解决 `file://` 协议下 fetch 跨域问题）。MPC / LP / MC 全部用 JavaScript 在浏览器本地计算，GitHub Pages 静态托管即可用，无需服务器。

## 三、目录结构

```
jepa-health-model/
├── server.py                  # Flask API 服务器（五层统一接口，本地完整版）
├── index.html                 # 根目录副本（GitHub Pages 部署入口）
├── data/
│   ├── db_layer.py            # L1: SQLite 存储层
│   ├── health_log.db          # 健康时序数据库
│   ├── jepa_model.pkl         # 训练好的 JEPA 权重
│   └── jepa_stats.json        # 归一化参数
├── models/
│   ├── jepa_model.py          # L2: JEPA 世界模型（NumPy MLP）
│   ├── export_weights.py      # 权重导出 JSON（供前端）
│   └── embed_weights.py       # 权重嵌入 HTML（解决 file:// CORS）
├── optimizer/
│   ├── mpc_optimizer.py       # L3: MPC 滚动优化（SLSQP）
│   └── mc_meals.py            # L4: 蒙特卡洛配餐（Python 版）
└── frontend/
    └── index.html             # L5: 单文件 SPA（含 JS 版 LP/MPC/MC + 119 种食物库）
```

## 四、快速开始

### 方式一：在线版（推荐，零依赖）

直接访问 <https://my990813.github.io/health-world-model/>。

### 方式二：本地完整版（Python 后端）

```bash
pip install flask flask-cors numpy scipy
python server.py
# 然后 POST http://localhost:5000/api/init 完成初始化
```

主要 API：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/init` | POST | 初始化数据库/食材/模拟历史 |
| `/api/logs` | GET | 查询近 N 天健康记录 |
| `/api/logs/add` | POST | 添加每日记录 |
| `/api/jepa/simulate` | POST | 模拟未来体重轨迹 |
| `/api/mpc/optimize` | POST | MPC 生成 7 天营养规划 |
| `/api/meals/generate` | POST | MC 生成当日三餐 |
| `/api/foods` | GET | 食材营养库 |
| `/api/dashboard` | GET | 仪表盘聚合数据 |

### 方式三：重训模型并更新前端权重

```bash
# 1. 训练（数据积累在 health_log.db 后执行训练入口）
# 2. 导出权重 JSON
python models/export_weights.py
# 3. 嵌入前端 HTML
python models/embed_weights.py
```

## 五、技术要点备忘

- **Big-M 数值稳定性**：LP 的 Big-M 取 1e4（曾用 1e6，在移动端 ARM 浮点下溢出导致手机配餐无输出）
- **异步链简化**：配餐按钮用单层 `setTimeout(0)`，避免移动端双 setTimeout 不触发
- **防御性设计**：用户未填写个人资料时，营养检查走通用阈值而非个性化计算
- **食物库**：119 种常见中式食材，覆盖主食面点 / 蔬菜 / 水果 / 蛋白 / 坚果五类，字段含 GI 与膳食纤维
