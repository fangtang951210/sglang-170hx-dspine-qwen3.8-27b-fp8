# 本地大模型推理速度测试工具 v2.2 → 170hx DSpine 测速填法(实测跑通版)

日期:2026-11-09 · 工具文件:`/Users/sugar/Documents/本地大模型推理速度测试工具v2.2 (1).html` · 服务:170hx sglang 0.5.22(DSpine 投测,端口 30020,未做任何改动)

## 一、逐字段填法(一字不差照抄)

| 输入框 | 填什么 |
|---|---|
| API 类型 | **OpenAI 兼容接口**(点选这个,不要选 Ollama) |
| API 地址 | `http://<LAN-IP>:30020/v1/chat/completions` ← **必须带完整路径 `/v1/chat/completions`,不能只填 `http://<LAN-IP>:30020`** |
| 模型名称 | `qwen3.8-27b` |
| API-Key | 留空(内网无鉴权,留空即可) |
| 备注 | 随意,如 `170hx CMP170HX qwen3.8-27b DSpine` |
| 最小提示词长度 | `8000` |
| 最大提示词长度 | `8000` |
| 步长 | `1000`(min=max 时只测一个点,步长值不影响) |
| 步长倍数 | `1` |
| 期望输出长度 | `512` |
| 并发数 | `1` |
| 超时 (ms) | `300000`(8K prefill + 512 输出约 10 秒,但默认 30000 偏紧,建议放大) |
| Temperature | `1.0` |
| Top-P | `0.1` |
| Presence / Frequency Penalty | `0` / `0` |

填完点 **开始测试**。

## 二、失败根因(已复现、已实锤)

用户失败原因 = **API 地址填了裸地址** `http://<LAN-IP>:30020`。此工具把输入框的值**原样**作为请求 URL,不自动拼接 `/v1/chat/completions`(源码默认值即含完整路径 `http://<LAN-IP>:8000/v1/chat/completions`)。裸地址 POST 到根路径 → sglang 返回 **405 Method Not Allowed** → 工具重试 3 次全败 → 页面报"失败 (所有 1 个并发请求都失败了)"。

排除项(都查过,均非根因):
- **CORS 不是问题**:sglang 0.5.22 自带 CORS 中间件,对 file:// 页面(Origin: null)的预检 OPTIONS 和实请求都正确返回 `access-control-allow-origin: null`、`access-control-allow-headers: authorization, content-type`,浏览器直接放行(在 file:// 页面上下文实测 fetch 200)。
- **模型名没问题**:`/v1/models` 返回 id 就是 `qwen3.8-27b`。

## 三、做了什么

只改了网页表单里的填法(URL 补全 + 超时放大),**服务端零改动**:DSpine 启动脚本、sglang 参数、服务进程均未动、未重启。测速请求本身即验证了服务正常。

## 四、实测结果(8K 单流,页面结果表原文)

| 提示词长度 | 预填充耗时(ms) | **预填充速度** | 输出长度 | 输出耗时(ms) | **输出速度** | 状态 |
|---|---|---|---|---|---|---|
| 8144(实际) | 3873.50 | **2102.49 tokens/s** | 695 | 6465.40 | **107.50 tokens/s** | 1/1 并发成功 |

注:实际 prompt 8144 tokens(比预期 8000 多 1.8%,工具按实际 token 计);输出 695 tokens 为 sglang usage 计数(含投机草稿接受 token)。

## 五、存证截图

- `/Users/sugar/Documents/tmp/2026-11-09-测速工具-整页结果.png`(参数区+结果表整页)
- `/Users/sugar/Documents/tmp/2026-11-09-测速工具-可视区结果.png`(结果表近景)

## 六、终态

DSpine 服务运行中,health 200,模型 `qwen3.8-27b` 在线。
