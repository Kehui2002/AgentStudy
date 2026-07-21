# Mini Agent 学习项目

这是一个从零实现 Agent 调用链的学习项目。第一阶段只实现最小的纯文本闭环：

```text
用户输入
  -> Agent.run()
  -> AgentLoop.run()
  -> ModelRequest(UserPromptPart)
  -> Model.request()
  -> ModelResponse(TextPart)
  -> TextOutputSchema.validate()
  -> AgentResult
```

## 当前已经实现

- 异步 `Agent.run()` 和同步包装 `Agent.run_sync()`
- 与 Provider 无关的 `Model` 抽象接口
- 可记录请求的 `FakeModel`
- `ModelRequest`、`ModelResponse`、`UserPromptPart`、`TextPart`
- 每次运行独立的 `AgentState` 和 `RunContext`
- 请求次数与运行步骤统计
- 纯文本输出校验
- 标准库 `unittest` 测试

`tools/`、结构化输出、节点状态机和真实 Provider 是后续阶段的扩展点，目前没有提前实现。

## 运行示例

当前环境不需要安装第三方依赖：

```bash
PYTHONPATH=src python3 examples/hello_agent.py
```

## 运行测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

如果以后安装了 `pytest`，也可以执行：

```bash
pytest
```

## 阅读顺序

1. `src/mini_agent/messages.py`：理解内部消息协议。
2. `src/mini_agent/models/base.py`：理解 Agent 与 Provider 的边界。
3. `src/mini_agent/models/fake.py`：理解如何在没有真实 API 的情况下测试。
4. `src/mini_agent/runtime/loop.py`：理解完整调用链。
5. `src/mini_agent/agent.py`：理解公开 API 如何隐藏内部实现。
6. `tests/test_text_run.py`：从外部行为反推内部设计。

## 下一阶段

加入 function tool 后，调用链将扩展为：

```text
ModelResponse(ToolCallPart)
  -> ToolManager 参数校验
  -> 执行 Python 函数
  -> ModelRequest(ToolReturnPart)
  -> 再次调用模型
```
