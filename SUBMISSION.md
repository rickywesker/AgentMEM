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
| Add API 地址 | `https://<railway-domain>/add` |
| Search API 地址 | `https://<railway-domain>/search` |
| 认证方式 | `Authorization: Token` |
| 记忆系统 Key | `.env` 里的 `AGENTMEM_API_KEY` |
| 公开 GitHub 仓库地址 | 推送后的仓库 URL |

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
> 3. **不做 LLM 事实抽取。** 实测对比六个抽取模型,弱模型明显有害,强模型与
>    完全不抽取持平,不值得付出 3–6 倍的写入耗时。相关代码保留在
>    `src/agentmem/extract.py`,默认关闭(`EXTRACT_API_BASE` 留空)。
>
> **合规**
>
> Search 只做候选的选择与排序,不合成任何新文本,也不读取问题的预期答案;返回的每
> 一条记录都可追溯到写入语料中的原始文本。`user_id` 隔离由 `store.load_user`
> 单点保证,无跨用户检索。仓库中不含任何评测集的答案、题目或 ID。
> 复现所用的离线 harness 在 `harness/`,其 answer/judge prompt 逐字取自贵方公开的
> 评测代码 `AML-memory/agent-memory-leaderboard`,已在 README 中署名。

## 提交前确认

- [ ] 仓库已推送且为 public
- [ ] Railway(或其他)部署完成,`https://<domain>/health` 从外网返回 200
- [ ] `scripts/stress.py` 对**公网地址**跑出 PASS
- [ ] `.env` 未进仓库
- [ ] 勾选「30 天内持续公网可访问且保持稳定」

## 一个需要向组委会确认的问题

表单写明:**「非工业榜申请中,Add / Search 过程中使用的模型必须为 gpt-4o-mini。」**

这条规则对生成式模型是清楚的,我们也已符合——默认配置在 Add/Search 全程不调用任何
生成式模型。但**向量检索用的 embedding 模型无法是 gpt-4o-mini**(它不提供 embedding
接口),规则是否覆盖 embedding 并不明确。

这不是一个可以回避的细节。三套配置都已实测(LoCoMo n=500):

| 配置 | 得分 | embedding | Add/Search 内的生成式模型 |
|---|---|---|---|
| embedding + 不抽取 | **62.2–63.4** | 用 | 无 |
| BM25 + gpt-4o-mini 抽取 | 57.8 | 不用 | gpt-4o-mini |
| 纯 BM25 | 55.6 | 不用 | 无 |

两种读法对应两套配置,差 5 分左右,都远超 ±1.4 的运行间方差:

- **embedding 不受限** → 用第一行,即当前默认配置,无需改动。
- **embedding 也必须是 gpt-4o-mini(即等价于禁用向量检索)** → 用第二行:
  `EMBED_API_BASE` 留空,`EXTRACT_API_BASE` 指向 gpt-4o-mini,
  `AGENTMEM_FACT_SHARE=0.5`。注意此时**不能**退回纯 BM25——没有 embedding 时,
  gpt-4o-mini 抽取值 +2.2 分,两者是部分冗余的替代关系。

切换只改环境变量,不改代码。

建议在邮件里直接问:

> 学术榜规则中「Add/Search 过程中使用的模型必须为 gpt-4o-mini」,是否包含用于向量
> 检索的 embedding 模型(如 text-embedding-3-small)?若包含,是否有指定的
> embedding 模型,还是要求完全不使用向量检索?
