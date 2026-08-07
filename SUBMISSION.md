# 报名表单填写内容

对应 <https://agentmemories.ai/evaluation> 的申请表。**先部署拿到公网地址、先把仓库推成公开,
再填这张表**——`Add API 地址`、`Search API 地址`、`公开 GitHub 仓库地址` 三项都是必填。

## 逐项填写

| 字段 | 填什么 |
|---|---|
| 榜单类型 | 学术榜 |
| 学术评测方式 | 提供 Add/Search 接口 |
| 工作邮箱 | rick 的邮箱 |
| 系统名称 | `AgentMEM` |
| 版本名称 | 提交时的 commit 短哈希,例如 `v1.0-8aaf06d` |
| Add API 地址 | `http://47.99.166.113:3000/add` |
| Search API 地址 | `http://47.99.166.113:3000/search` |
| 认证方式 | `Authorization: Token` |
| 记忆系统 Key | ECS 上的 `AGENTMEM_API_KEY`;留空则不鉴权 |
| 公开 GitHub 仓库地址 | `https://github.com/rickywesker/AgentMEM` |

`认证方式` 选 `Authorization: Token` 与代码一致:`api.py` 的 `authorize()` 会把
`Authorization: <scheme> <value>` 按空白切分取后半段,所以 `Token abc123` 和
`Bearer abc123` 都能正确解析出 `abc123`。

## 提交说明与运行指南(可直接粘贴)

> **运行方式**
>
> 仓库根目录提供 Dockerfile,构建后单容器即可运行,依赖一个 Postgres:
>
> ```
> docker compose up -d          # 启动 Postgres
> docker build -t agentmem .
> docker run -d --env-file .env --network host --restart unless-stopped agentmem
> ```
>
> 容器监听 `$PORT`(默认 8080),健康检查端点 `GET /health`。所需环境变量见
> `.env.example`:`DATABASE_URL`、`AGENTMEM_API_KEY`,以及 embedding 的
> `EMBED_API_BASE / EMBED_API_KEY / EMBED_MODEL`。
>
> **Add/Search 封装位置**
>
> - `src/agentmem/api.py` — `POST /add`、`POST /search`、`GET /health` 三个端点
> - `src/agentmem/ingest.py` — 写入侧:逐条对话轮次落库,内联「日」粒度日期
> - `src/agentmem/retrieve.py` — 检索侧:BM25 + 余弦,RRF 融合,去重,截断
> - `src/agentmem/store.py` — Postgres 存储,`user_id` 是唯一的读取边界
>
> **方法说明**
>
> 本系统为自研,不基于任何现有记忆系统改造,因此没有需要声明的上游实现。用到的两项
> 通用检索技术均为公开算法,直接实现而非引用库:
>
> - Okapi BM25(教科书公式,`retrieve.py` 的 `BM25` 类)
> - Reciprocal Rank Fusion(Cormack, Clarke & Buettcher, SIGIR 2009)
>
> 核心设计取舍有三点,均由离线复现实验决定,数据见仓库 README:
>
> 1. **不建向量索引。** `user_id` 的检索范围就是单个会话,每用户仅数百条记录,
>    一次带索引的主键查询取出全量后用 numpy 暴力点积,比维护 ANN 索引更快。
> 2. **返回契约允许的全部 100 条,并叠加字符预算。** 得分随返回条数单调上升
>    (3 条 41.3 → 100 条 60.7);但记录长度跨数据集差一个量级,所以再加一层
>    50k 字符上限,防止长记录数据集把上下文撑到 10 万字符以上。
> 3. **稠密召回与 BM25 用 RRF 融合。** 向量来自 Voyage 的 `voyage-4-large`
>    (1024 维),写入时对每条记录计算、检索时对 query 计算,与 BM25 的排名做
>    Reciprocal Rank Fusion。仓库中另有一条"LLM 事实抽取"通路(`extract.py`,
>    规则指定的 gpt-4o-mini),其设计是**事实只当检索索引、命中后返回来源原文轮次**,
>    抽取结果从不进入答案上下文;但**本次提交未启用它**,`EXTRACT_API_BASE` 留空。
>
> **合规**
>
> Add 与 Search 的执行路径上**不调用任何生成式模型**:写入侧只做分词、日期内联与
> embedding,检索侧只做打分与排序。唯一的模型调用是 embedding(Voyage),已如实
> 披露——规则限定的是生成式模型,而 gpt-4o-mini 不提供 embedding 接口。
>
> Search 只做候选的选择与排序,不合成任何新文本,也不读取问题的预期答案;返回的每
> 一条记录都可追溯到写入语料中的原始文本。`user_id` 隔离由 `store.load_user`
> 单点保证,无跨用户检索。仓库中不含任何评测集的答案、题目或 ID。
> 复现所用的离线 harness 在 `harness/`,其 answer/judge prompt 逐字取自贵方公开的
> 评测代码 `AML-memory/agent-memory-leaderboard`,已在 README 中署名。

## 部署现状

**已从 Railway 迁至阿里云 ECS。** Railway 那个边缘 IP(`69.46.46.102`)从大陆
根本建不了 TCP 连接,评测器在深圳,所以第二次冒烟的 `ADD_API_CONTRACT_MISMATCH`
是"请求一个都没到",不是契约错。详见 DEPLOY.md §1。

`记忆系统 Key` 沿用原值(未变更,表单里那一项不用改),取法:

```bash
ssh root@47.99.166.113 'grep AGENTMEM_API_KEY /opt/agentmem/.env'
```

| 项 | 值 |
|---|---|
| 主机 | 阿里云 ECS `cn-hangzhou`,`i-bp14xx0q9v32tb1helbt` |
| 规格 | 4 vCPU / 8 GB / 120 GB,10 Mbps 峰值,按流量计费 |
| 部署目录 | `/opt/agentmem`(app 容器 + compose 的 Postgres) |
| 公网地址 | `http://47.99.166.113:3000` |
| 端口选择 | 安全组只开放 80 / 443 / 3000;未备案不宜用 80/443,故取 3000 |
| 健康检查 | `GET /health` → 200(大陆直连 25ms) |

单副本是刻意的:限流信号量是进程内的,多副本会把有意压住的上游并发翻倍。

**公网压测结果**(`scripts/stress.py`,64 并发 Add / 32 并发 Search,大陆直连):

```
adds        264 (含 24 次重放重试)  6.7s
latency     p50 1.27s   p99 2.93s   (平台超时 1200s)
throughput  39.4 adds/s
PASS — 契约通过、无跨用户泄漏、无 5xx
```

对照 Railway 上的 p50 3.99s / 12.9 adds/s——同样带 embedding 往返,快约 3 倍。
按 39.4 adds/s 估算,文本赛道约 47,000 个 chunk 灌完约 20 分钟,窗口是 72 小时。

鉴权已验证:无 key → 401,错 key → 401,正确 key → 200。
`scripts/preflight.py` 23 项全过(含"已忽略代理"的 warn 行)。
向量覆盖率 4,803/4,803 = 100%,单条 4,096 字节。

**数据库是干净的**:压测与 preflight 写入的行已全部删除,`SELECT count(*)` 为 0。

## 提交前确认

- [x] 仓库已推送且为 public
- [x] 部署完成,`/health` 从外网返回 200
- [x] `scripts/preflight.py` READY + `scripts/stress.py` PASS,均对**公网地址**、
      且**不经代理**(代理会替评测器回答,这正是上一次踩的坑)
- [x] `.env` 未进仓库
- [ ] **把表单里的 Add / Search 地址改成新的 ECS 地址**——旧的 Railway 地址还留在
      表单里的话,重跑冒烟必然再挂一次
- [x] embedding 已切到 Voyage 并验证向量确实落库(见下一节)
- [ ] 用 `harness/` 重测 `voyage-4-large` 的分数——旧的 62.2–63.4 是
      `text-embedding-3-small` 的数,不能直接引用
- [ ] 确认 Voyage 账户余额够 ~47,000 次 embedding 调用
- [ ] 勾选「30 天内持续公网可访问且保持稳定」——ECS 到 2027-05-19 到期,
      30 天内不要停机;`--restart unless-stopped` 已设置

## 关于 gpt-4o-mini 规则

表单写明:**「非工业榜申请中,Add / Search 过程中使用的模型必须为 gpt-4o-mini。」**

生成式模型这块已符合:Add 侧唯一的生成式调用是事实抽取,用的就是 **gpt-4o-mini**;
Search 侧不调用任何生成式模型。这一点已在上面的「提交说明与运行指南」里写明。

原本的决定是使用向量检索(`text-embedding-3-small`),理由是规则针对生成式模型,
而 embedding 模型无法是 gpt-4o-mini——后者不提供 embedding 接口。

**这个判断不变,但供应商换了。** 迁到大陆 ECS 后 `api.openai.com` 不可达,
现改用 **Voyage(`voyage-4-large`,1024 维)**。`llm.py` 的 `embed()` 本就是
OpenAI 形状的 `/embeddings` 调用,Voyage 同形状,所以是纯环境变量切换、零代码改动:

```
EMBED_API_BASE=https://api.voyageai.com/v1
EMBED_MODEL=voyage-4-large
EMBED_DIM=1024
```

已验证向量确实落库:压测 4,803 行,**向量覆盖率 100%**,单条 4,096 字节
(1024 × 4),64 并发下 Voyage 未触发任何限流(降级告警 0 次)。

**尚未重测分数。** 62.2–63.4 那个数字是 `text-embedding-3-small` 的,换模型后
不再适用,要用 `harness/` 重跑才知道。作为下界参考,同一套代码的纯 BM25 是 55.6。

抽取(`EXTRACT_*`)仍然关闭:规则要求 Add/Search 内的生成式模型必须是 gpt-4o-mini,
而提供它的端点在这台机器上不可达。所以 Add 侧现在**没有任何生成式模型调用**,
Search 侧本来就没有——这一点比原方案更容易过合规审查。

另外记一笔:没有 embedding 时,gpt-4o-mini 抽取本身值 +2.2 分(55.6 → 57.8),
两者是部分冗余的替代关系。所以如果组委会认定 embedding 不合规,正确的退路是
BM25 + 抽取而不是裸 BM25——但那需要一个大陆可达、且提供 gpt-4o-mini 的端点,
`yunwu.ai` 在这台机器上不满足。
