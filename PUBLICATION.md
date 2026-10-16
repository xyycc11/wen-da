# EchoMind 发布说明

2026-10-16。基于已有EchoMind项目的个人学习与改造版本，主要补充MySQL用户画像、Milvus适配、工具可靠性与验证。不声称所有上游功能从零原创；未提供新的整仓开源授权声明。

## 当前实现

- FastAPI对话API；LLM/Embedding/规则三路意图识别；主辅Agent编排；工具白名单、业务Skills与MCP接入。
- 查询改写、并行检索及重排；缓存、超时、熔断、降级；Prometheus与回复评测。
- Redis保存近期消息、摘要和回执，MySQL保存JSON用户画像，Milvus保存知识库和情景记忆。
- MySQL使用独立用户主键、短事务、版本CAS、候选JSON校验和线程池调用；模型生成期间不持有数据库事务。
- Milvus使用本地ONNX模型、COSINE检索、元数据归属过滤及集合配置校验；模型准备需明确执行。

## 启动前必读

1. 创建Python3.12环境，安装requirements；按`.env.example`填写本地私有`.env`，不要上传真实密码或Token。
2. 提供Redis、MySQL及Milvus连接。`MYSQL_PROFILE_URL`必须为`mysql+pymysql`数据库URL，密码特殊字符需要URL编码。
3. 在独立开发数据库运行`python -m tooling.init_profile_db`创建画像表。运行`python -m tooling.prepare_vector_model`显式准备模型资产。
4. 完成[MySQL指南](wiki/MYSQL_PROFILES.md)及[Milvus指南](wiki/MILVUS_MIGRATION.md)的数据库配置后，才启动`python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`。旧安装暂不迁移画像时需明确使用`PROFILE_BACKEND=legacy_vector`。

MySQL覆盖文件需要独立应用/Root密码，主机端口3307；容器应用URL使用profile-mysql主机，而本地应用使用127.0.0.1及实际映射端口。先启动依赖、建表和验证，再启动应用，不盲目执行全量启动并认为数据已迁完。

原Chroma兼容代码和迁移工具保留以支持旧数据，不是新画像存储。Milvus旧指南中的user_profile集合仅用于兼容旧数据，当前默认新画像进入MySQL。

## 验证与边界

开发目录最近一次完整离线回归：162通过、1条既有Pydantic类型警告。MySQL测试通过注入临时SQLite引擎验证SQL/事务，并编译MySQL DDL；Milvus测试使用假客户端及假向量。依赖检查和Compose配置合并通过。

真实MySQL/Milvus联合部署、真实ONNX推理与模型质量、实际旧数据迁移及生产负载尚未完成验收。部署配置能解析不等于容器已健康启动，离线测试不是线上成功率。Redis/MySQL/Milvus不是分布式事务，不承诺跨库原子提交。发布副本复测结果以发布交付记录为准。

不包含真实.env、数据卷、会话、日志、虚拟环境、凭据、私人学习知识库或简历。安全筛查只是发布前检查，不是全面安全审计。发布副本复测结果：162项通过，1条既有类型定义警告；无真实模型/数据库调用。
