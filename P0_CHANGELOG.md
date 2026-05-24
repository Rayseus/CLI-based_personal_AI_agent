# P0 变更记录：确定性验证增强 + 关闭默认会话日志

> 日期：2026-05-24  
> 对应 `DEVELOPMENT_PLAN.md` §4.3 / §5 中尚未完全落地的两条防幻觉规则。

## 背景与思路

MVP 阶段 `verify_citations` 只做一件事：剥离数据库中不存在的 `[memory:id]`。  
计划里还有两道**确定性**防线未实现：

1. **无引用声称**：回复暗示「根据记忆 / you told me」却没有合法 citation → 退化为中性回复  
2. **空 search 硬边界**：`search_memory` 返回 `[]` 后，模型仍编造答案或乱引 id → 强制拒答

设计原则保持不变：**验证是后处理中间件，不暴露给 LLM，不可被 prompt 绕过。**

### 关键设计决策

| 决策 | 理由 |
| --- | --- |
| 新增 `verify_reply()`，保留 `verify_citations()` | 旧单测与 API 兼容；agent 层走完整策略 |
| 在 `chat()` 内跟踪 `allowed_memory_ids` | citation 必须来自**本轮** `search_memory` 结果，而非任意已存在 id |
| 空 search 且无拒答措辞 → 整句替换为固定模板 | 比「删 citation 留正文」更严，符合 §5 第 4 条 |
| 无引用声称 → `NEUTRAL_REPLY` 而非 `UNKNOWN_REPLY` | 区分「检索为空」与「有记忆但无法核实引用」 |
| 默认关闭 `agent.log` 写入 | 用户要求停止记录 CLI 与 agent 的通话/诊断轨迹 |

## 做法

### 1. `tools.py` — `verify_reply()`

```
用户提问 → search_memory → 记录 allowed_memory_ids
                ↓
         LLM 生成最终文本
                ↓
    _strip_invalid_citations()
      · id 不在 DB → fail，剥离
      · id 不在 allowed_memory_ids → fail_not_in_search，剥离
                ↓
    allowed_memory_ids == ∅ 且无拒答措辞？
      → YES: 返回 UNKNOWN_REPLY + log empty_search_override
                ↓
    无合法 citation 且匹配 _UNCITED_CLAIM_RE？
      → YES: 返回 NEUTRAL_REPLY + log uncited_claim
                ↓
         返回 cleaned 文本
```

**正则（节选）**

- 拒答识别：`_ADMISSION_RE` — `don't know` / `haven't told me` / `不知道` 等  
- 无引用声称：`_UNCITED_CLAIM_RE` — `you told me` / `based on what you` / `根据记忆` 等  
- **刻意排除** save 确认语（如 `Got it, I noted that.`），避免误杀

### 2. `agent.py` — 接入与日志

- 每轮 tool loop 中，`search_memory` 返回后更新 `allowed_memory_ids`  
- 最终回复走 `verify_reply(..., allowed_memory_ids=...)`  
- 新增 `_configure_diagnostics()` / `_log_diagnostics()`：仅 `AGENT_DEBUG=1` 或 `AGENT_LOG_ENABLED=1` 时输出 `[TOOL]` / `[VERIFY]`  
- **默认不再**打开 `agent.log`、不再把 stderr 重定向到文件

## 改动文件

| 文件 | 变更 |
| --- | --- |
| `tools.py` | 新增 `verify_reply`、`UNKNOWN_REPLY`、`NEUTRAL_REPLY`、正则与辅助函数 |
| `agent.py` | 跟踪 `allowed_memory_ids`；调用 `verify_reply`；默认关闭诊断日志 |
| `tests/test_tools.py` | 新增 `TestVerifyReply`（6 用例） |
| `tests/test_agent.py` | `TestChatLoop` 启用 diagnostics；更新空 search 伪造 citation 用例 |
| `.env.example` | 新增 `AGENT_LOG_ENABLED=0`，注释说明诊断开关 |

## 测试

```bash
python -m unittest discover tests -v
```

新增覆盖：

- 空 search + 编造答案 → `UNKNOWN_REPLY`  
- 空 search + 正确拒答 → 保留原文  
- citation id 存在但不在 search 结果 → 剥离  
- citation id 在 search 结果 → 保留  
- 无 citation 的 memory 声称 → `NEUTRAL_REPLY`  
- save 确认语不误触发

## 遇到的问题

### 1. 空 search 后「删 citation 留正文」不够

原 `verify_citations` 去掉 `[memory:99]` 后可能剩下 `"Your favorite color is blue."`，仍属幻觉。  
**解决**：空 search 分支整句 override，除非已含拒答措辞。

### 2. id 在 DB 但不在 search 结果

旧逻辑 `exists(id)` 为真即通过；模型可跳过 search 直接引用历史 id。  
**解决**：引入 `allowed_memory_ids`，search 未返回的 id 一律 `fail_not_in_search`。

### 3. 单测捕获不到 `[TOOL]` 日志

关闭默认日志后，`chat()` 内 `_log_diagnostics` 默认静默。  
**解决**：`TestChatLoop.setUp` 调用 `_configure_diagnostics(True)`。

### 4. save 确认 vs memory 声称

"You told me..." 与 "I noted that" 语义不同。  
**解决**：`_UNCITED_CLAIM_RE` 只匹配**检索引用**类措辞，不匹配 save 确认。

## 如何重新开启诊断日志

```bash
# 终端可见
AGENT_DEBUG=1 python agent.py

# 仅写入文件
AGENT_LOG_ENABLED=1 python agent.py
```

两者可同时开启（debug 模式下终端 + 文件）。
