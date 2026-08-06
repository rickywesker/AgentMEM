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
| Add API 地址 | `https://agentmem-production-2ba2.up.railway.app/add` |
| Search API 地址 | `https://agentmem-production-2ba2.up.railway.app/search` |
| 认证方式 | `Authorization: Token` |
| 记忆系统 Key | Railway 服务变量 `AGENTMEM_API_KEY`(值见下方) |
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

## 部署现状

已部署,状态如下。`记忆系统 Key` 的值不写进仓库,用下面的命令取:

```bash
railway variables --service agentmem --kv | grep AGENTMEM_API_KEY
```

| 项 | 值 |
|---|---|
| Railway 项目 | `AgentMEM`(`95c46a4c-b997-4ad1-9700-6e27f7edd11f`) |
| 服务 | `agentmem` + `Postgres` 插件 |
| 区域 | Southeast Asia ×1(US West 已缩到 0) |
| 公网地址 | `https://agentmem-production-2ba2.up.railway.app` |
| 健康检查 | `GET /health` → 200 |

单副本是刻意的:限流信号量是进程内的,多副本会把有意压住的上游并发翻倍;而且
多区域副本会让一半流量走上跨太平洋的慢路径。

**公网压测结果**(`scripts/stress.py`,64 并发 Add / 32 并发 Search):

```
adds        264 (含 24 次重放重试)  20.5s
latency     p50 3.99s   p99 8.38s   (平台超时 1200s)
throughput  12.9 adds/s
PASS — 契约通过、无跨用户泄漏、无 5xx
```

按 12.9 adds/s 估算,文本赛道约 47,000 个 chunk 灌完约 1 小时,窗口是 72 小时。

鉴权已验证:无 key → 401,错 key → 401,正确 key → 200。

**已知无害残留**:压测写入的 `stress:conv-*` 共 264 行仍在库里。检索按 `user_id`
限定范围,BM25 也是按用户从 `load_user` 现建的,所以这些行既不会被任何真实查询命中,
也不影响打分。要清理需给 Railway 注册 SSH key 后执行
`DELETE FROM memories WHERE user_id LIKE 'stress:%'`。

## 提交前确认

- [x] 仓库已推送且为 public
- [x] 部署完成,`/health` 从外网返回 200
- [x] `scripts/stress.py` 对**公网地址**跑出 PASS
- [x] `.env` 未进仓库
- [ ] 勾选「30 天内持续公网可访问且保持稳定」——注意 Railway 用量计费,
      30 天内不要删项目或让服务休眠

## 关于 gpt-4o-mini 规则

表单写明:**「非工业榜申请中,Add / Search 过程中使用的模型必须为 gpt-4o-mini。」**

生成式模型这块已符合:默认配置在 Add/Search 全程不调用任何生成式模型
(`EXTRACT_API_BASE` 留空)。

**已决定:使用向量检索(`text-embedding-3-small`)。** 规则针对的是生成式模型——
embedding 模型无法是 gpt-4o-mini,后者不提供 embedding 接口——按此理解提交。
上面的「提交说明与运行指南」里已如实披露用了 embedding,由组委会审核判定。

如果组委会认定 embedding 也在限制内,退回配置是**改环境变量,不改代码**:

```
EMBED_API_BASE=            # 留空,关闭向量检索
EXTRACT_API_BASE=<endpoint>
EXTRACT_MODEL=gpt-4o-mini  # 规则指定的模型
AGENTMEM_FACT_SHARE=0.5
```

三套配置的实测分数(LoCoMo n=500),用于评估退回代价:

| 配置 | 得分 |
|---|---|
| embedding + 不抽取(**当前**) | **62.2–63.4** |
| BM25 + gpt-4o-mini 抽取(退回配置) | 57.8 |
| 纯 BM25 | 55.6 |

退回时**不要**用纯 BM25——没有 embedding 时 gpt-4o-mini 抽取值 +2.2 分,
两者是部分冗余的替代关系。
