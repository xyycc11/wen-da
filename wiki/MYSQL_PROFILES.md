# MySQL用户画像

2026-10-16实现：Redis保存近期消息/摘要，Milvus保存知识库/情景记忆，MySQL保存结构化偏好与实体。MySQL画像不做Embedding，不参与相似度检索。

## 数据与并发边界

`echomind_user_profiles` 表：`user_key`（用户ID的SHA256主键）、`user_id`、`profile` JSON（preferences列表及entities字典）、`version`、`source_conv_id`、UTC `updated_at`。哈希主键避免MySQL默认不区分大小写的排序规则混淆Alice与alice，取出后还检查原用户ID。

每次操作独立连接，写入使用SQLAlchemy短事务，异常回滚；先读取版本，等待模型生成结束后以`WHERE version=expected_version`写入，旧版本冲突丢弃而不盲目重试。首次插入主键冲突也不覆盖。模型请求期间不持有SQL事务。Redis画像锁仍保留，数据库版本约束作为额外保护。后台失败只记错误类型，不输出数据库URL或完整异常参数。

画像候选只用用户消息和已有画像，结构/条数/长度/大小均校验，不接受权限等额外顶层字段。仍可能有语义误提取，需要真实模型评测和隐私策略，不能声称消除提示注入或完全准确。用户ID必须来自已鉴权调用端。

## 启用

安装项目requirements。创建独立开发数据库和最小权限应用用户，将URL写入本地私有环境（不要上传）：

```dotenv
PROFILE_BACKEND=mysql
MYSQL_PROFILE_URL=mysql+pymysql://echomind:URL_ENCODED_PASSWORD@127.0.0.1:3306/echomind?charset=utf8mb4
VECTOR_BACKEND=milvus
```

密码特殊字符需要URL编码；不要在聊天里发送密码。先明确建表，然后启动应用：

```powershell
python -m tooling.init_profile_db
```

该命令仅在已配置数据库创建专用表，**不会创建数据库或删除其他表**。正常请求不自动建表；API及CLI启动检查表可访问，MySQL不可用或配置缺失会报错，不静默改用SQLite或向量画像。上线时建表权限可用独立管理员、运行用户只授所需读写权限。

本地MySQL不可用时可选择`docker-compose.mysql.yml`独立覆盖：需要`ECHOMIND_MYSQL_PASSWORD`、`ECHOMIND_MYSQL_ROOT_PASSWORD`两个独立密码及指向容器`profile-mysql:3306`的MYSQL_PROFILE_URL。主机端口3307避免抢占已有3306。先只启动profile-mysql，再通过一次性应用容器运行建表命令，最后启动应用。与原Compose/Milvus覆盖文件一起使用；本轮不自动启动或改私人.env。Compose配置和真实镜像运行是两项不同验收。

## 旧画像与回滚

默认画像后端现为mysql，原向量画像数据不删除，也不自动迁入。旧安装如果暂不具备MySQL，必须显式设置`PROFILE_BACKEND=legacy_vector`继续旧读写；新路径不读取或双写向量画像。

已有画像迁入：暂停旧画像写入，导出完整user_profile集合（保留user_id/document/metadata），用`python -m tooling.import_profiles_mysql --input <source.json>`先只读检查，再加`--apply --ack-source-paused`。支持上次向量迁移工具导出的source.json（含user_profile键）；同用户按updated_at/ts选择最新，先验证全部记录。目标已有不同内容会报冲突而非覆盖，目标相同内容可重复执行；源文件和原库保留。跨用户批量导入不是一个大事务，失败时已导入行保留，需按报告检查后续跑，不自动切换。

回滚前停写并核对新MySQL画像；直接切回旧向量库不会自动带回新偏好。本次不提供反向同步，不删除任何源。

## 验证口径

SQLite仅作为测试注入引擎，执行真实SQL/事务验证JSON往返、用户隔离、版本冲突及异常回滚；同时编译MySQL方言DDL。不能把这些测试称为真实MySQL验收。真实数据库连接/建表/重启持久性/认证以及真实模型质量需在独立开发库联调；未完成时简历只写“实现MySQL存储与离线验证”，不写生产效果或真实迁移完成。

SQLAlchemy行为参考：[MySQL方言](https://docs.sqlalchemy.org/en/20/dialects/mysql.html)、[事务与会话](https://docs.sqlalchemy.org/en/20/orm/session_basics.html)。
