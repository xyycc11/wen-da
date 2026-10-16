# EchoMind：Chroma → Milvus 实施与迁移说明

日期：2026-10-14。当前交付仅代码、离线测试与配置校验，不代表真实服务或旧数据迁移已完成。

## 一、现在的存储边界

- Redis：近期消息、摘要、既有会话锁，逻辑不改；情景记忆持久化失败时，不裁剪原始消息或替换摘要。
- Milvus `knowledge_base`：资料分块与向量检索。
- Milvus `episodic`：带 `user_id`、`conv_id` 的情景记忆；检索必须指定用户，沿用当前会话优先/同用户历史兜底逻辑。
- Milvus `user_profile`：按 ID/用户元数据标量读取。向量字段只有无语义单位占位值；禁止语义检索，不需要调用 Embedding。
- 不新增 MySQL，不修改 EchoWorld、私有 `.env` 或已有数据。Chroma 依赖、服务和卷暂时保留以支持迁移及旧模式。

内部管理用的 `get()` 可以全量读取，专供迁移等可信调用；用户隔离仍必须由经过鉴权的调用端提供用户 ID，不能把原始适配器暴露给玩家或模型。

## 二、改动入口

| 文件 | 作用 |
|---|---|
| `core/vector_store.py` | 后端工厂；安全等值/AND过滤；Milvus增改删查及分页；集合模型签名/维度/指标校验 |
| `core/vector_embedding.py` | 独立本地384维MiniLM ONNX；遮罩均值池化、归一化、有限值与维度检查；模型文件指纹 |
| `tooling/knowledge_base.py` | 保留知识库外部接口，存储转到适配层 |
| `memory/conversation_memory.py` | 保留记忆接口；确定性情景ID；持久化失败不裁剪Redis；画像归属核验 |
| `tooling/prepare_vector_model.py` | 显式准备模型，下载校验SHA256及安全解包；请求路径不自动下载 |
| `tooling/migrate_chroma_to_milvus.py` | 默认只读清点；明确apply才导出备份、重新向量化、校验目标及源数据 |
| `api/main.py` | 生命周期关闭向量客户端 |
| `docker-compose.milvus.yml` | 新增Milvus/etcd/MinIO服务，覆盖应用后端，不改原Compose文件 |
| `tests/test_milvus_store.py` | 假客户端离线契约、隔离、失败保护、迁移和配置不匹配测试 |

采用 `FLAT + COSINE`，先以精确检索降低原型复杂度，未做ANN性能优化。Milvus相似度转为兼容层的 `distance=1-score`，知识库返回分数即余弦相似度。旧Chroma若用默认L2，分数不可直接比较，阈值必须重新校准。

## 三、本地配置（尚未执行切换）

在项目根目录、已安装项目依赖的环境中运行：

```powershell
python -m tooling.prepare_vector_model
```

这一步会访问网络下载模型；本轮没有执行。可设置 `VECTOR_MODEL_DIR` 指向已备好的 `model.onnx` 和 `tokenizer.json` 所在目录。代码不静默切到付费API或另一个模型；更换权重、分词器或维度，需要新集合前缀并重新迁移。

切换时使用这些配置（不要现在覆盖已有私有配置）：

```dotenv
VECTOR_BACKEND=milvus
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION_PREFIX=echomind_migration_20261014
MILVUS_TOKEN=
```

旧安装未设置 `VECTOR_BACKEND` 时默认 `chroma`；新 `.env.example` 明确示范 `milvus`。Milvus连接失败或集合不匹配会报错，绝不偷偷回退Chroma。旧Chroma模式保留原HTTP失败转本地路径逻辑，所以迁移源必须明确指定，不能猜数据在哪。

## 四、真实迁移步骤（需要后续启动Docker）

以下只是操作说明，本轮未执行。先确认备份空间及目标前缀未被其他环境使用。原应用运行期间不要直接启动覆盖配置的应用服务。

1. 启动Docker Desktop，只启动新增数据库依赖：

```powershell
docker compose -f docker-compose.yml -f docker-compose.milvus.yml up -d milvus-etcd milvus-minio milvus
```

2. 明确原数据来源并只读清点。HTTP示例使用原Compose的8001映射；本地示例必须是含 `chroma.sqlite3` 的实际目录。三个源集合需要已存在。

```powershell
python -m tooling.migrate_chroma_to_milvus --source-kind http --source-host localhost --source-port 8001 --target-prefix echomind_migration_20261014
# 若实际是本地持久库，改用这一条（替换路径）：
python -m tooling.migrate_chroma_to_milvus --source-kind local --source-path C:/path/to/actual/chroma --target-prefix echomind_migration_20261014
```

默认不建立Milvus目标，只输出数量和内容哈希。不得把示例路径直接当成真实源。

3. 暂停所有源写入者（应用、副本、后台任务），保留Chroma服务供只读导出。确认静止后才执行：

```powershell
python -m tooling.migrate_chroma_to_milvus --source-kind http --source-host localhost --source-port 8001 --target-prefix echomind_migration_20261014 --apply --ack-source-paused
```

`--ack-source-paused` 只是操作者确认，不会自动替你停服务。脚本先把文档/元数据导出到本地 `data/migration_backups/<唯一目录>/source.json`，再写目标；**这不是完整物理备份，不含原始向量和索引**，原Chroma卷/目录也必须保留备份。包含私人会话信息的备份不得上传GitHub。

目标必须为空或只含完全相同的迁移记录，否则拒绝覆盖。写完逐集合比较完整记录哈希，再重新检查源是否变化；成功写 `manifest.json`，仅表示“已核验暂存”，不会自动切换或删除源。失败后保留部分目标和备份，查明原因后可用同前缀重复相同数据，或另选新前缀。源持续写入时不能据此做一致性承诺。

4. 用真实服务验证：中文业务问题召回、分块更新与旧块清理、同用户跨会话、跨用户拒绝、画像读取、Milvus断连时Redis原消息保留、重启后持久性。内容哈希通过不等于召回质量通过；检查相似度阈值后再切换。

5. 本地运行修改对应环境变量并重启应用；容器运行明确指定同一目标前缀后，再构建/启动覆盖配置中的 `echomind` 服务。覆盖配置仍保留原Chroma依赖以便回滚。不要执行删除卷的命令。

回滚：停写→确认切换后是否有新增数据→必要时导出/核对并补齐→显式设回 `VECTOR_BACKEND=chroma` 与正确源连接→重启。**切换后有新增写入时，直接改回旧库会丢失这些新记录的可见性，不能声称无损回滚。**本轮没有实现反向增量同步。

## 五、已验证与未验证

- 本轮隔离Python3.12环境，安装 `pymilvus==2.6.17`；依赖检查通过。
- 标准SDK集合Schema描述及相关方法签名已离线核对。
- `docker compose -f docker-compose.yml -f docker-compose.milvus.yml config --quiet` 实际退出码0；仅说明配置可合并，不证明镜像拉取或健康启动。
- 测试使用确定性假向量及假Milvus/Redis/模型，无付费接口。完整测试结果见本地知识库交付记录。
- 没启动Docker、没运行真实Milvus、没下载/运行真实ONNX模型、没构建镜像、没迁移真实记录、没改私有配置、没发布GitHub。
- Compose只绑定回环端口，MinIO是开发示例凭据，不适合直接生产上线；生产认证、资源限制、备份与恢复演练另做。
- Redis与Milvus不是分布式事务；本次仅确保长期写失败时不主动裁剪短期数据。知识库upsert/旧块delete也不是原子文档替换，进程内锁不是跨副本事务。
- 原有情景记忆正文截断策略仍在，迁移无法补回原本未保存的内容。

## 六、参考

- [Milvus查询迭代器API](https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Vector/query_iterator.md)：分页接口。
- [Milvus集合描述API](https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Collections/describe_collection.md)：检查集合描述和字段。
- [Milvus官方发布记录](https://github.com/milvus-io/milvus/releases)：部署版本参考，镜像运行仍待实机验证。
- [Chroma 0.5.23默认ONNX模型实现](https://raw.githubusercontent.com/chroma-core/chroma/0.5.23/chromadb/utils/embedding_functions/onnx_mini_lm_l6_v2.py)：旧默认模型及预处理参考；原库实际Embedding若不同，必须先确认，不可假定兼容。
