# ZCF量化炒股 - 基于强化学习的股票交易系统

## 🚀 项目现状（2025年12月更新）

经过多次的迭代和优化，项目已发展成为一个功能完善的**实时股票交易预测系统**，支持多版本模型、实时数据源、LLM市场情报、以及最新推出的**V11全功能集成版本**。

## 📊 最新版本：V11 全功能集成版

**V11**是当前最新版本，整合了V7、V9、V10的所有功能，实现多模型协同工作和智能融合决策。

### V11核心特性

- ✅ **多模型融合**：PPO + LSTM + Transformer + 全息模型协同决策

- ✅ **实时可视化**：Web端实时图表展示（端口8082）

- ✅ **持仓编辑器**：网页端实时修改持仓状态（端口5001）

- ✅ **智能融合决策**：多模型预测结果加权融合

- ✅ **完整功能集成**：技术指标、LLM解释、时间序列、多模态处理

### 快速使用V11

```bash

cd ZCF_RL_STOCK

python real_time_predict_v11.py

```

然后访问：

- 实时可视化：http://127.0.0.1:8082

- 持仓编辑器：http://127.0.0.1:5001

## 🎯 版本演进

| 版本 | 观察空间 | 核心特性 | 推荐场景 | 状态 |

|------|---------|---------|---------|------|

| **V7** | 126维价格序列 | 纯技术分析，稳定可靠 | 新手入门，稳健投资 | ✅ 稳定 |

| **V8** | 29维（21技术+8LLM） | 技术+市场情报，功能全面 | 综合决策，进阶使用 | ✅ 稳定 |

| **V9** | 历史窗口+组合+LLM | LSTM/GRU、注意力机制、动态参数优化 | 高级用户，组合管理 | ✅ 稳定 |

| **V10** | Transformer架构 | Transformer、多模态、实时可视化 | 前沿技术探索 | ✅ 稳定 |

| **V11** | 多模型融合 | **全功能集成，智能融合决策** | **推荐使用** | ⭐ **最新** |

### 版本详情

#### V7 - 纯技术分析版

- **观察空间**：126维价格序列

- **特点**：纯技术分析，稳定可靠

- **回测收益**：+10.57%

- **适用场景**：新手入门，稳健投资

#### V8 - LLM增强版

- **观察空间**：29维（21技术指标 + 8维LLM情报）

- **特点**：集成LLM市场情报，7维度市场分析

- **适用场景**：综合决策，需要市场情报支持

#### V9 - 时间序列深度学习版

- **特点**：LSTM/GRU、注意力机制、动态参数优化、自动学习优化

- **适用场景**：高级用户，需要时间序列模式识别

#### V10 - Transformer前沿版

- **特点**：Transformer模型、多模态数据处理、实时可视化、全息动态模型

- **适用场景**：前沿技术探索，需要长期依赖捕捉

#### V11 - 全功能集成版 ⭐

- **特点**：整合V7+V9+V10所有功能，多模型智能融合

- **模型权重**：PPO 40% + LSTM 20% + Transformer 20% + 全息 20%

- **适用场景**：推荐所有用户使用

## 📖 监督学习与强化学习的区别

监督学习（如 LSTM）可以根据各种历史数据来预测未来的股票的价格，判断股票是涨还是跌，帮助人做决策。

而强化学习是机器学习的另一个分支，在决策的时候采取合适的行动 (Action) 使最后的奖励最大化。与监督学习预测未来的数值不同，强化学习根据输入的状态（如当日开盘价、收盘价等），输出系列动作（例如：买进、持有、卖出），使得最后的收益最大化，实现自动交易。

## 🤖 系统架构

### 动作空间

所有版本统一使用 **7个离散动作**：

- 0: 卖出 100%

- 1: 卖出 50%

- 2: 卖出 25%

- 3: 持有

- 4: 买入 25%

- 5: 买入 50%

- 6: 买入 100%

### 奖励函数

奖励函数基于持仓收益，考虑交易成本和风险控制：

```python

# 当前利润

reward = current_net_worth - initial_balance

# 风险惩罚

if drawdown > threshold:

    reward -= penalty

```

## 🚀 快速开始

### 1. 环境安装

```bash

# 克隆项目

git clone <repository_url>

cd ZCF-RLSTOCK

# 安装依赖

pip install -r ZCF_RL_STOCK/requirements.txt

```

### 2. 数据准备

```bash

cd ZCF_RL_STOCK

# 使用 baostock（免费，推荐新手）

python get_stock_data_v7_myportfolio.py

# 或使用 Tushare（需要注册，支持实时数据）

python get_stock_data_v7_tushare.py

# 或使用 AkShare（免费，支持实时数据）

python get_stock_data_v7_akshare.py

```

### 3. 模型训练（可选）

```bash

# 训练 V7 模型

python train_v7.py

```

或直接使用预训练模型（推荐）：

- `ppo_stock_v7.zip`

- `models_v7/best/best_model.zip`

### 4. 实时预测

```bash

# 使用 V11 版本（推荐，全功能集成）

python real_time_predict_v11.py

# 或使用 V7 版本（稳定可靠）

python real_time_predict_v7_600730.py

# 或使用 V9 版本（时间序列深度学习）

python real_time_predict_v9.py

# 或使用 V10 版本（Transformer）

python real_time_predict_v10.py

```

## 📊 核心功能

### 1. 实时预测系统

- ✅ 多数据源支持（Tushare/AkShare/baostock）

- ✅ 自动操作记录和汇总

- ✅ 实时持仓信息显示

- ✅ LLM 市场情报参考

- ✅ 实时可视化（V10/V11）

- ✅ 网页持仓编辑器（V7/V11）

### 2. LLM 市场情报

提供 7 个维度的市场分析：

1. ✅ **宏观经济数据** - GDP、CPI、利率政策分析

2. ✅ **新闻和舆情分析** - 市场热点、负面消息识别

3. ✅ **市场情绪指标** - 恐慌指数 VIX、投资者情绪

4. ✅ **资金流向数据** - 外资、融资融券、北向资金

5. ✅ **政策变化信息** - 货币/财政/监管政策影响

6. ✅ **国际市场联动** - 美股、港股相关性分析

7. ✅ **突发事件应对** - 地缘政治、疫情、自然灾害

### 3. 操作记录系统

- 自动记录所有仓位变动

- 操作汇总和待执行列表

- 历史操作查询

- CSV 格式日志文件

### 4. 持仓管理

- 支持手动更新持仓状态

- 网页端实时修改持仓信息（V7/V11）

- 自动加载和保存持仓信息

- 同步外部交易记录

### 5. 实时可视化（V10/V11）

- Web端实时图表展示

- 价格走势图

- 技术指标图表

- 综合仪表板

- 访问地址：http://127.0.0.1:8082

### 6. 多模型融合决策（V11）

- PPO模型（40%权重）

- LSTM/GRU模型（20%权重）

- Transformer模型（20%权重）

- 全息动态模型（20%权重）

- 智能加权融合生成最终决策

## 📁 项目结构

```

ZCF-RLSTOCK/

├── ZCF_RL_STOCK/              # 主代码目录

│   ├── rlenv/                 # 强化学习环境

│   ├── train_v7.py            # V7 训练脚本

│   ├── real_time_predict_v11.py  # V11 实时预测（推荐）

│   ├── real_time_predict_v10.py  # V10 实时预测

│   ├── real_time_predict_v9.py   # V9 实时预测

│   ├── real_time_predict_v7_600730.py  # V7 实时预测

│   ├── llm_market_intelligence.py  # LLM 市场情报

│   ├── lstm_gru_time_series.py  # V9 LSTM/GRU模块

│   ├── transformer_model.py   # V10 Transformer模块

│   ├── multimodal_data_processor.py  # V10 多模态处理

│   ├── realtime_visualization.py  # V10/V11 可视化

│   ├── holographic_dynamic_model.py  # V10 全息模型

│   ├── models_v7/             # V7 模型文件

│   └── stockdata_v7/          # 股票数据

├── docs/                      # 文档目录

│   ├── 快速开始.md

│   ├── 训练指南.md

│   ├── 实时预测.md

│   ├── LLM市场情报.md

│   ├── 持仓管理.md

│   ├── 版本/

│   │   ├── V7版本说明.md

│   │   ├── V8版本说明.md

│   │   └── V9版本说明.md

│   └── ...

├── models_v7/                 # 训练好的模型

├── stockdata_v7/              # 股票数据

└── README.md                  # 本文件

```

## 📚 文档导航

详细文档请查看 [docs/README.md](docs/README.md)：

- [🚀 快速开始](docs/快速开始.md) - 5分钟上手

- [🎯 训练指南](docs/训练指南.md) - 模型训练教程

- [📈 实时预测](docs/实时预测.md) - 实时预测系统

- [💼 实盘交易](docs/实盘交易.md) - 实盘交易指南

- [💡 LLM 市场情报](docs/LLM市场情报.md) - LLM 集成说明

- [💼 持仓管理](docs/持仓管理.md) - 持仓状态管理

- [📥 数据获取](docs/数据获取.md) - 数据源配置

- [🐛 问题排查](docs/问题排查.md) - 常见问题解决

- [📊 版本对比](docs/版本对比.md) - 各版本功能对比

## 🕵️‍♀️ 验证结果

### V7 模型回测

- **初始本金**: 100,000 元

- **股票代码**: `sh.600036` (招商银行)

- **回测收益**: +10.57%

- **模型稳定性**: 高

### 实时预测效果

系统支持实时数据获取和预测，能够：

- 实时获取最新股票数据

- 自动执行交易操作

- 记录所有操作历史

- 提供市场情报参考

- 多模型协同决策（V11）

## ⚙️ 配置说明

### 环境变量

```bash

# DeepSeek API 密钥

export DEEPSEEK_API_KEY='your_api_key'

# Tushare Token（可选）

export TUSHARE_TOKEN='your_token'

```

### 主要配置

```python

# real_time_predict_v11.py

MODEL_PATH = "ppo_stock_v7.zip"  # 模型路径

STOCK_CODE = 'sh.600036'  # 股票代码

ENABLE_LLM = True  # 是否启用 LLM

ENABLE_LSTM_PREDICTION = True  # 是否启用 LSTM

ENABLE_TRANSFORMER = True  # 是否启用 Transformer

ENABLE_MULTIMODAL = True  # 是否启用多模态

ENABLE_VISUALIZATION = True  # 是否启用可视化

VISUALIZATION_PORT = 8082  # 可视化端口

ENABLE_WEB_EDITOR = True  # 是否启用持仓编辑器

WEB_EDITOR_PORT = 5001  # 持仓编辑器端口

```

## 🔧 数据源对比

| 数据源 | 实时数据 | 费用 | 需要注册 | 推荐度 |

|--------|---------|------|---------|--------|

| Tushare | ✅ | 免费/付费 | ✅ | ⭐⭐⭐⭐⭐ |

| AkShare | ✅ | 免费 | ❌ | ⭐⭐⭐⭐ |

| baostock | ❌ | 免费 | ❌ | ⭐⭐⭐ |

## ⚠️ 重要提示

1. **仅供学习研究**: 本系统仅供学习和研究使用，不构成投资建议

2. **风险自负**: 实盘交易存在风险，请谨慎操作，风险自负

3. **数据准确性**: 请确保数据源可靠，定期验证数据准确性

4. **模型局限性**: 模型基于历史数据训练，无法预测突发事件

## 🛠️ 工具脚本

### 持仓更新工具

```bash

# 交互式更新

python update_portfolio.py

# 命令行更新

python update_portfolio.py --stock sh.600036 --shares 500 --balance 80000 --price 43.25

# 查看当前状态

python update_portfolio.py --show

```

## 📈 版本历史

- **V7** (2024): 126维价格序列，纯技术分析，稳定可靠

- **V8** (2024): 集成 LLM 市场情报，29维特征空间

- **V9** (2024): LSTM/GRU时间序列处理，注意力机制，动态参数优化

- **V10** (2025): Transformer模型，多模态处理，实时可视化，全息动态模型

- **V11** (2025): **全功能集成版，多模型智能融合决策** ⭐

## 👻 最后

- 股票 Gym 环境主要参考 [Stock-Trading-Environment](https://github.com/notadamking/Stock-Trading-Environment)，对观测状态、奖励函数和训练集做了修改。

- **⚠️ 重要**: 本系统仅供学习和研究使用，不构成投资建议。实盘交易需谨慎，风险自负。

### 开源项目

- [notadamking/Stock-Trading-Environment](https://github.com/notadamking/Stock-Trading-Environment)

- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

- [baostock](http://baostock.com/) - 免费证券数据平台

- [Tushare](https://tushare.pro/) - 金融数据接口

- [AkShare](https://akshare.akfamily.xyz/) - 金融数据接口库

### 教程和文档

- [Create custom gym environments from scratch — A stock market example](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e)

- [Welcome to Stable Baselines docs! - RL Baselines Made Easy](https://stable-baselines3.readthedocs.io/)

- [PPO 算法论文](https://arxiv.org/abs/1707.06347)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

详见 [LICENSE](LICENSE) 文件

---

**最后更新**: 2025-12-01  

**当前推荐版本**: V11 全功能集成版  

**推荐模型**: V7 (稳定可靠) 或 V11 (功能最全)
