# Mini Agent 学习项目

这是一个从零实现 Agent 调用链的学习项目。当前已经支持纯文本与 Function Tool
闭环：

```text
用户输入
  -> Agent.run()
  -> AgentLoop.run()
  -> ModelRequest(UserPromptPart)
  -> Model.request()
  -> ModelResponse(ToolCallPart)
  -> Pydantic 参数校验
  -> 执行 Python Function Tool
  -> ModelRequest(ToolResultPart)
  -> Model.request()
  -> ModelResponse(TextPart)
  -> TextOutputSchema.validate()
  -> AgentResult
```

## 当前已经实现

- 异步 `Agent.run()` 和同步包装 `Agent.run_sync()`
- 与 Model Provider 无关的 `Model` 抽象接口
- 可记录请求的 `FakeModel`
- 通过异步 HTTP 调用真实 API 的 `DeepSeekModelProvider`
- `ModelRequest`、`ModelResponse`、`UserPromptPart`、`TextPart`
- `ToolCallPart`、`ToolResultPart` 与多轮串行工具调用
- 从同步或异步 Python 函数签名生成 JSON Schema
- 使用 Pydantic 严格校验嵌套模型、集合、枚举、可选值、联合参数和结构化返回值
- 参数错误回传模型纠正，以及独立的工具重试上限
- 可恢复 `ToolError` 安全回传模型，意外工具异常脱敏并中止运行
- 每次运行独立的 `AgentState` 和 `RunContext`
- 模型请求、工具调用、执行和纠错次数统计
- 纯文本输出校验
- 标准库 `unittest` 测试

工具并行执行、结构化 Agent 输出、流式响应和节点状态机是后续阶段的扩展点。

## 运行示例

安装项目依赖：

```bash
python3 -m pip install -e .
```

运行不访问外部 API 的示例：

```bash
PYTHONPATH=src python3 examples/hello_agent.py
```

使用真实 DeepSeek API 运行 Function Tool 闭环：

```bash
set -a
source .env
set +a
PYTHONPATH=src python3 examples/deepseek_tool_agent.py
```

项目根目录的 `.env` 已被 Git 忽略，应包含：

```dotenv
DEEPSEEK_API_KEY=你的真实Key
RUN_DEEPSEEK_INTEGRATION=1
```

API Key 只通过环境变量提供，不要写入源码或提交到 Git。DeepSeek 工具调用当前
显式关闭思考模式。

## 运行测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

加载 `.env` 后可单独运行真实集成测试：

```bash
PYTHONPATH=src python3 -m unittest tests.test_deepseek_tool_integration -v
```

如果以后安装了 `pytest`，也可以执行：

```bash
pytest
```

## 阅读顺序

1. `src/mini_agent/messages.py`：理解内部消息协议。
2. `src/mini_agent/models/base.py`：理解 Agent 与 Model Provider 的边界。
3. `src/mini_agent/models/fake.py`：理解如何在没有真实 API 的情况下测试。
4. `src/mini_agent/models/deepseek.py`：理解内部消息与 Model Provider API 的转换。
5. `src/mini_agent/runtime/loop.py`：理解完整调用链。
6. `src/mini_agent/agent.py`：理解公开 API 如何隐藏内部实现。
7. `src/mini_agent/tools/`：理解工具注册、Schema 生成、校验和执行。
8. `tests/test_tool_run.py` 和 `tests/test_deepseek_model.py`：从公开行为反推设计。

## 下一阶段

- 异步和并行 Function Tool
- 工具执行超时与人工批准
- DeepSeek 思考模式下的 `reasoning_content`
- 结构化输出和流式响应

## Origin Fit 本地批准流程

`origin-fit` 在 Linux 本地导入 UTF-8 CSV，将完整数据保存到内容寻址文件存储，
并只通过 `inspect` 暴露有界 Dataset Summary。以下命令使用仓库中的确定性
ExpDec2 合成数据演示 `import → inspect → propose → approve`：

```bash
origin-fit --state-dir .origin-fit import tests/fixtures/synthetic_expdec2.csv \
  --x time_s --y decay_a --y decay_b --y decay_c \
  --uncertainty decay_a=decay_a_error \
  --uncertainty decay_b=decay_b_error \
  --uncertainty decay_c=decay_c_error \
  --unit time_s=s \
  --unit decay_a=dimensionless --unit decay_b=dimensionless \
  --unit decay_c=dimensionless \
  --unit decay_a_error=dimensionless \
  --unit decay_b_error=dimensionless \
  --unit decay_c_error=dimensionless

origin-fit --state-dir .origin-fit inspect '<dataset-snapshot-id>'

origin-fit --state-dir .origin-fit propose '<dataset-snapshot-id>' \
  --experiment-id synthetic-expdec2 --fit-min 0 --fit-max 11 \
  --weighting instrument --initialization origin-auto \
  --graph-profile expdec2-standard@1.0

origin-fit --state-dir .origin-fit inspect '<fit-specification-id>'
ORIGIN_FIT_OPERATOR=researcher@example.test \
  origin-fit --state-dir .origin-fit approve '<fit-specification-id>'
origin-fit --state-dir .origin-fit inspect '<approved-fit-recipe-id>'
```

`propose` 的 Fit Specification 始终没有执行权限。只有独立的 `approve` CLI
操作会创建 Approved Fit Recipe；任意语义变化都会产生新的规格哈希并要求重新批准。

## 远程 Origin Worker 与 Fit Archive

安装 Worker 可选依赖后，可以前台方式启动仅用于开发和 Linux 回归测试的
Fake Origin Worker：

```bash
python3 -m pip install -e '.[origin-worker]'
export ORIGIN_WORKER_TOKEN='至少 32 个字符的部署秘密'
origin-worker serve \
  --state-dir .origin-worker \
  --host 192.168.56.1 \
  --port 8443 \
  --certfile /受控路径/origin-worker.crt \
  --keyfile /受控路径/origin-worker.key \
  --fake-origin
```

Worker 拒绝 `0.0.0.0`/`::` 通配监听，并要求 HTTPS 证书与 Bearer Token；部署时还应
用 Windows 防火墙把监听地址限制到 Linux 虚拟机。Linux 客户端通过
`HttpWorkerTransport.with_pinned_certificate(...)` 固定校验自签名证书，随后由
`RemoteOriginExecutor.execute_approved_fit(...)` 隐藏 capability 协商、幂等提交、
状态轮询、Bundle 下载、manifest/文件哈希校验和 Fit Archive 持久化。

`/v1` 提供 `health`、`capabilities`、幂等任务提交、状态、取消和 Bundle 下载接口。
Worker SQLite 保存队列和状态元数据，Dataset Snapshot 与 Bundle 保存在逐任务隔离
工作区；Linux SQLite 保存 Worker Job 映射和归档索引，内容寻址对象存储长期保存
Snapshot 与验证后的 Bundle。可用现有命令查看本地索引：

```bash
origin-fit --state-dir .origin-fit inspect '<fit-job-id>'
origin-fit --state-dir .origin-fit inspect '<fit-archive-id>'
```

`--fake-origin` 只生成确定性的测试结果和占位 PNG/PDF/OPJU 产物，不代表真实
OriginPro 自动化已经验证；真实 OriginPro 覆盖仍必须在显式启用的 Windows 验收环境
中完成。
