# NLLB 翻译服务

基于 Meta NLLB-200 模型的本地翻译服务，使用 OpenVINO 加速推理。

## 项目结构

```
nllb-win/
├── run_server.py               # 一键启动入口
├── requirements.txt            # Python 依赖
├── .gitignore
├── README.md
└── src/
    ├── __init__.py
    ├── config.py               # 配置文件（模型路径、设备、语言映射）
    ├── download_model.py       # 下载 HuggingFace 模型
    ├── export_model.py         # 导出 OpenVINO 模型
    ├── detector.py             # 语言检测
    ├── translator.py           # 翻译引擎
    ├── server.py               # Flask Web 服务器
    └── templates/
        └── index.html          # 翻译页面
```

## 快速开始

### 前提条件

- Python 3.8 或更高版本
- 网络连接（仅首次下载模型需要）

### 启动

```bash
python run_server.py
```

**首次运行**会自动完成：
1. 创建虚拟环境 `.venv/`
2. 安装所有依赖
3. 下载 NLLB 模型（~2.6GB）
4. 编译为 OpenVINO 模型
5. 启动 Web 服务器

**后续运行**检测到模型已存在，直接启动。

启动后浏览器访问 **http://localhost:5000**

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | 翻译页面 |
| `POST` | `/translate` | 翻译文本 |
| `GET` | `/languages` | 支持的语言列表 |

### 翻译示例

```bash
curl -X POST http://localhost:5000/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","source_lang":"auto"}'
```

## 配置

编辑 `src/config.py`：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `DEVICE` | `GPU` | 推理设备（`CPU` / `GPU` / `AUTO`） |
| `TARGET_LANGUAGE` | `zho_Hans` | 目标语言（简体中文） |
| `PORT` | `5000` | 服务端口（在 run_server.py 中修改） |

## 支持的语言

| 语言 | code |
|---|---|
| 自动检测 | `auto` |
| 英文 | `en` |
| 日文 | `ja` |
| 韩文 | `ko` |
| 法文 | `fr` |
| 德文 | `de` |
| 中文 | `zh-cn` |
