import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import create_engine, event, update
from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable

from memory.profile_store import MySQLProfileStore, ProfileConflict, validate_profile
from memory.conversation_memory import MemoryManager, Message, MsgRole


@pytest.fixture
def store(tmp_path):
    engine = create_engine('sqlite:///' + str(tmp_path / 'profiles.sqlite3'),
                           connect_args={'check_same_thread': False})
    target = MySQLProfileStore(engine=engine)
    target.create_schema()
    yield target
    target.close()


def profile(text='简洁回复'):
    return {'preferences': [text], 'entities': {'产品': ['产品A']}}


def test_profile_roundtrip_restart_and_owner_isolation(store):
    assert store.load('Alice') is None
    assert store.save('Alice', profile(), 'conv1', expected_version=0) == 1
    assert store.load('alice') is None
    assert store.load('Bob') is None
    reopened = MySQLProfileStore(engine=store.engine)
    assert reopened.load('Alice').data == profile()
    assert reopened.load('Alice').source_conv_id == 'conv1'
    assert store.save('Alice', profile('英文'), 'conv2', expected_version=1) == 2
    assert store.load('Alice').data == profile('英文')


def test_user_id_is_bound_parameter_not_sql(store):
    user = "u' OR 1=1 --"
    store.save(user, profile(), 'c', expected_version=0)
    assert store.load(user).data == profile()
    assert store.load('u') is None


@pytest.mark.parametrize('value', [None, [], {'role': 'admin'}, {'preferences': '中文'},
    {'preferences': [1]}, {'preferences': ['']}, {'entities': []},
    {'entities': {'产品': 'A'}}, {'preferences': ['a'] * 33}, {'preferences': ['a' * 257]}])
def test_invalid_profile_preserves_previous_record(store, value):
    store.save('u', profile(), 'c', expected_version=0)
    with pytest.raises(ValueError):
        store.save('u', value, 'c', expected_version=1)
    assert store.load('u').version == 1
    assert store.load('u').data == profile()


def test_stale_update_and_duplicate_create_do_not_overwrite(store):
    store.save('u', profile(), 'c1', expected_version=0)
    with pytest.raises(ProfileConflict):
        store.save('u', profile('旧结果'), 'c2', expected_version=0)
    store.save('u', profile('新结果'), 'c3', expected_version=1)
    with pytest.raises(ProfileConflict):
        store.save('u', profile('旧结果'), 'c2', expected_version=1)
    assert store.load('u').data == profile('新结果')


def test_transaction_rolls_back_after_update_error(store):
    store.save('u', profile(), 'c', expected_version=0)
    def fail_after_update(conn, cursor, statement, params, context, many):
        if statement.lstrip().upper().startswith('UPDATE'):
            raise RuntimeError('simulated post-statement failure')
    event.listen(store.engine, 'after_cursor_execute', fail_after_update)
    try:
        with pytest.raises(RuntimeError):
            store.save('u', profile('新结果'), 'c', expected_version=1)
    finally:
        event.remove(store.engine, 'after_cursor_execute', fail_after_update)
    assert store.load('u').data == profile()
    assert store.load('u').version == 1


def test_corrupt_owner_is_not_returned(store):
    store.save('u', profile(), 'c', expected_version=0)
    with store.engine.begin() as connection:
        connection.execute(update(store.table).values(user_id='other'))
    with pytest.raises(ValueError, match='owner'):
        store.load('u')


def test_mysql_schema_compiles_with_json_and_innodb(store):
    sql = str(CreateTable(store.table).compile(dialect=mysql.dialect()))
    assert 'JSON' in sql and 'InnoDB' in sql and 'utf8mb4' in sql


@pytest.mark.parametrize('url', ['', 'sqlite:///bad.sqlite', 'mysql://u:p@localhost/db', 'not-a-url'])
def test_bad_configuration_never_falls_back(monkeypatch, url):
    monkeypatch.delenv('MYSQL_PROFILE_URL', raising=False)
    with pytest.raises(ValueError):
        MySQLProfileStore(url)


def manager(store, response=None):
    memory = MemoryManager.__new__(MemoryManager)
    memory._profile_store = store
    memory._profile = Mock()
    memory._get_working_memory = AsyncMock(return_value=[
        Message(MsgRole.USER, '请简洁回复'), Message(MsgRole.ASSISTANT, '虚构偏好不要使用')])
    memory._model = 'fake'
    memory._client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=
        SimpleNamespace(content=[{'type': 'text', 'text': json.dumps(response or profile())}]))))
    return memory


def test_memory_reads_and_writes_mysql_without_vector_profile(store):
    memory = manager(store)
    asyncio.run(memory._update_profile('u', 'c'))
    assert asyncio.run(memory._get_profile('u')) == profile()
    assert asyncio.run(memory._get_profile('other')) == {}
    memory._profile.upsert.assert_not_called()
    memory._profile.get.assert_not_called()
    prompt = memory._client.messages.create.call_args.kwargs['messages'][0]['content']
    assert '虚构偏好不要使用' not in prompt


def test_profile_read_failure_does_not_generate_or_overwrite():
    broken = SimpleNamespace(load=Mock(side_effect=RuntimeError('private error')), save=Mock())
    memory = manager(broken)
    asyncio.run(memory._update_profile('u', 'c'))
    memory._client.messages.create.assert_not_called()
    broken.save.assert_not_called()
    assert asyncio.run(memory._get_profile('u')) == {}
    memory._profile.get.assert_not_called()


def test_no_transaction_held_over_model_and_stale_model_result_is_discarded(store):
    store.save('u', profile(), 'c', expected_version=0)
    memory = manager(store)
    async def generate(**kwargs):
        assert store.engine.pool.checkedout() == 0
        store.save('u', profile('较新偏好'), 'other', expected_version=1)
        return SimpleNamespace(content=[{'type': 'text', 'text': json.dumps(profile('较旧偏好'))}])
    memory._client.messages.create.side_effect = generate
    asyncio.run(memory._update_profile('u', 'c'))
    assert store.load('u').data == profile('较新偏好')
    assert store.load('u').version == 2


def test_invalid_model_output_keeps_old_profile(store):
    store.save('u', profile(), 'c', expected_version=0)
    memory = manager(store, {'preferences': 'invalid'})
    asyncio.run(memory._update_profile('u', 'c'))
    assert store.load('u').version == 1


def test_truncated_model_output_keeps_old_profile(store):
    store.save('u', profile(), 'c', expected_version=0)
    memory = manager(store)
    memory._client.messages.create.return_value.stop_reason = 'max_tokens'
    asyncio.run(memory._update_profile('u', 'c'))
    assert store.load('u').version == 1


def test_new_manager_requests_only_episode_vector_collection(monkeypatch, store):
    import memory.conversation_memory as module
    factory = Mock(return_value=(Mock(), {'episodic': Mock()}))
    monkeypatch.setattr(module, 'build_collections', factory)
    monkeypatch.setattr(module, 'AsyncAnthropic', Mock())
    monkeypatch.setattr(module.redis, 'from_url', Mock())
    memory = MemoryManager(profile_store=store, profile_backend='mysql')
    assert factory.call_args.args[0] == ['episodic']
    assert memory._profile is None
    asyncio.run(memory.initialize())


def test_import_selects_latest_profile_and_is_idempotent(store):
    from tooling.import_profiles_mysql import prepare_profiles, import_profiles
    rows = prepare_profiles({'user_profile': [
        {'document': json.dumps(profile('old')), 'metadata': {'user_id': 'u', 'ts': '2026-10-01T00:00:00'}},
        {'document': json.dumps(profile('new')), 'metadata': {'user_id': 'u', 'ts': '2026-10-02T00:00:00'}}]})
    assert import_profiles(rows, store) == {'verified': 1, 'inserted': 1}
    assert store.load('u').data == profile('new')
    assert import_profiles(rows, store) == {'verified': 1, 'inserted': 0}


def test_import_preflight_conflict_prevents_other_writes(store):
    from tooling.import_profiles_mysql import import_profiles
    store.save('existing', profile(), 'c', expected_version=0)
    with pytest.raises(ProfileConflict):
        import_profiles([('new', profile(), 'c'), ('existing', profile('different'), 'c')], store)
    assert store.load('new') is None


def test_import_rejects_missing_owner_and_ambiguous_timestamp():
    from tooling.import_profiles_mysql import prepare_profiles
    with pytest.raises(ValueError):
        prepare_profiles([{'document': '{}', 'metadata': {}}])
    with pytest.raises(ValueError, match='ambiguous'):
        prepare_profiles([{'document': json.dumps(profile(text)), 'metadata': {'user_id': 'u'}}
                          for text in ['a', 'b']])
