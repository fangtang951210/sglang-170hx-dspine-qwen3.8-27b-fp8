# DSpine 草稿模型下载验收报告

- 日期:2026-11-09
- 目标机:170hx(ssh 别名 `170hx`)
- 目标目录:`/mnt/storage/models/llm/qwen3.8-27b-dspine-b16`
- 范围:仅下载与验证,未起服务、未动 sglang、未改任何现有配置

---

## 一、查证结论(先查证再动手)

**官方 repo 全名:`tencent/Qwen3.8-27B-DSpine-b16`——控制台第二次下载用的名字是正确的,无需纠正。**

依据:GitHub 官方 README(github.com/Tencent/DSpine)Supported Models 表原文:

| Target model | Draft model |
| --- | --- |
| `Qwen3-8B` | tencent/Qwen3-8B-DSpine-b16 |
| `Qwen3.8-27B` | **tencent/Qwen3.8-27B-DSpine-b16** |

HF 页面(huggingface.co/tencent/Qwen3.8-27B-DSpine-b16/tree/main)与 HF API(`/api/models/.../tree/main?recursive=true`)逐文件比对,官方口径 **5 个文件、共约 4.05 GB**(十进制),与此前调研口径(约 4.05G、5 个文件)一致:

| 文件 | 官方大小(HF API,字节) | 存储 |
| --- | --- | --- |
| .gitattributes | 1,519 | git |
| LICENSE | 11,544 | git |
| README.md | 10,628 | git |
| config.json | 2,302 | git |
| model.safetensors | 4,051,831,880 | xet |
| **合计** | **≈4,051,847,875(≈4.05 GB)** | |

注:该模型为**单文件权重**(单个 model.safetensors),HF 上本就没有 `model.safetensors.index.json`,验证时按"无 index 属正常"处理。

## 二、现场清点结果(只看不动)

- **现存下载进程 2 个**(控制台第二次 nohup 下载的遗留):
  - PID **13491**:`bash -c` 包装层(nohup 启动器)
  - PID **13493**:`~/sglang-llm-venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(\"tencent/Qwen3.8-27B-DSpine-b16\", local_dir=\"/mnt/storage/models/llm/qwen3.8-27b-dspine-b16\", max_workers=8)"` ← 真正的下载器
- 目录当时状态:4 个小文件已齐(config.json/.gitattributes/LICENSE/README.md,字节数与官方一致),model.safetensors 在 `.cache` 中下载中,目录 1.6G→1.8G 持续增长(观测三次递增),下载健康。
- 日志 `~/deploy/dspine-download.log`:停在 `Fetching 5 files: 80%|████| 4/5`(即只剩大文件),仅一条无 HF_TOKEN 的限速提示,无报错。
- 第一次 huggingface-cli 旧命令的失败**未留任何进程/文件残留**。
- 磁盘:`//<LAN-IP>/storage` 19T 总量、**可用 15T(21% 已用)**——2026-11-02 巡查记录的"模型盘 100% 满"现状已不成立,4G 模型无压力。

## 三、收编决策与理由

**决策:不杀不重下,等现存进程(PID 13493)自然完成后直接验证。**

理由(对照单子的两种情形):
1. 13493 下载的 repo(`tencent/Qwen3.8-27B-DSpine-b16`)与官方 README 完全一致 → 属于"现存进程下载的 repo 正确"情形;
2. 进程活着、目录体积持续增长、日志无错 → 下载健康,杀掉重下纯属浪费;
3. 因此走"轮询等待"分支:远端每 15 秒查进程存活与目录大小,14:05:19 进程自然退出,日志输出 `Fetching 5 files: 100%|██████████| 5/5` + `DONE /mnt/storage/models/llm/qwen3.8-27b-dspine-b16`。全程 HF 直连(总耗时约 9.5 分钟),**未动用 hf-mirror 镜像**。

未发生需要精确 kill 的情形(未用 pkill -f)。

## 四、最终验证结果

**结论:模型完整,sha256 内容级校验通过,现场干净。**

1. **文件清单与体积**:5 个文件逐一字节数与 HF API 完全一致:

   | 文件 | 本地(字节) | 官方(HF API) | 一致 |
   | --- | --- | --- | --- |
   | .gitattributes | 1,519 | 1,519 | ✓ |
   | LICENSE | 11,544 | 11,544 | ✓ |
   | README.md | 10,628 | 10,628 | ✓ |
   | config.json | 2,302 | 2,302 | ✓ |
   | model.safetensors | 4,051,831,880 | 4,051,831,880 | ✓ |

   `du -sh` 显示 3.8G(二进制 GiB;4,051,831,880 B = 十进制 4.05 GB,与 HF 页显示一致)。
2. **sha256 内容校验(非仅体积)**:
   - 本地:`sha256sum model.safetensors` = `ac90f1df842ea66b23e4224f3e86624add547526f689ffe9e9c6b7a5937116b1`
   - HF API `lfs.oid`(即文件内容的 sha256)= `ac90f1df842ea66b23e4224f3e86624add547526f689ffe9e9c6b7a5937116b1`
   - **完全一致,权重内容级校验通过**。小文件为 git 存储(非 LFS),按字节数比对。
3. **config.json 可解析**:`json.load` 正常,`model_type=qwen3`,`architectures=['DFlashDraftModel']`(DSpine 草稿模型架构,合理);safetensors index 不存在属正常(单文件权重)。
4. **现场清洁**:
   - 下载进程(13491/13493)均已退出,`ps aux | grep -E "snapshot_download|huggingface|hf download"` 为空;
   - 无 `.incomplete` 残留;`.cache/huggingface/download/` 下 huggingface_hub 留下的 5 个空 `.lock` flock 文件已删除(进程已退出、内容为空,删除零风险);
   - 保留 `.cache/huggingface/` 内 8 个 `.metadata`/`trees` 元数据文件(hf 标准增量缓存,下次 snapshot_download 校验用,不碍任何加载)。

**异常与说明**:无功能性异常。两条备忘——①旧告警"170hx 模型盘满"现状已不成立(现为 15T 可用);②下载为无 token 直连,日志有一条限速提示但实际速度足够(4G/9.5min),未换镜像。

**完成证据**:目录 5 文件齐全、逐字节+sha256 双口径对上官方;无下载进程、无 .incomplete/.lock 残留;本报告含上述四段。
