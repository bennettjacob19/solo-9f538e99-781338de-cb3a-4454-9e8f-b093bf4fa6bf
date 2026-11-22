"""端口状态区间查询：区间语义、reboot 复位、同毫秒行序、索引构建生命周期、并发共用。"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import jobs as jobs_module
from app import portindex as portindex_module
from app.config import Settings
from app.main import create_app
from app.portindex import IndexManager, PortIndex, build_index
from app.sorter import SortError

BASE = 1_795_000_000_000  # 约 2026-11 的 epoch 毫秒


# ---------- 工具 ----------

def make_settings(tmp_path: Path, quota_bytes: int = 1 << 20,
                  batch_bytes: int = 128, max_workers: int = 4) -> Settings:
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return Settings(
        logs_dir=logs,
        temp_dir=tmp_path / "tmp",
        results_dir=tmp_path / "results",
        quota_bytes=quota_bytes,
        batch_bytes=batch_bytes,
        max_workers=max_workers,
        ts_field="ts_ms",
    )


def write_log(logs_dir: Path, name: str, events: list) -> None:
    text = "".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events)
    (logs_dir / name).write_text(text, encoding="utf-8")


def run_job(client: TestClient, filename: str, timeout: float = 20.0) -> str:
    r = client.post("/api/jobs", json={"filename": filename})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = client.get(f"/api/jobs/{job_id}").json()
        if snap["state"] == "completed":
            return job_id
        assert snap["state"] != "failed", snap["error"]
        time.sleep(0.02)
    raise AssertionError(f"作业 {job_id} 未在 {timeout}s 内完成")


def query_intervals(client: TestClient, job_id: str, switch: str, port: str,
                    start: int, end: int, timeout: float = 20.0) -> list:
    """轮询直到索引就绪，返回区间列表（202=构建中、409=失败已触发重试，均继续等）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}/port-status",
                       params={"switch": switch, "port": port,
                               "start_ms": start, "end_ms": end})
        if r.status_code in (202, 409):
            time.sleep(0.02)
            continue
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["switch"] == switch and body["port"] == port
        assert body["start_ms"] == start and body["end_ms"] == end
        return body["intervals"]
    raise AssertionError(f"索引未在 {timeout}s 内就绪")


def iv(start: int, end: int, state: str) -> dict:
    return {"start_ms": start, "end_ms": end, "state": state}


# ---------- 区间语义 ----------

def test_intervals_basic_and_carry_in(tmp_path):
    """未知→掉线→在线；起点前事件纳入；其它端口/交换机互不影响。"""
    settings = make_settings(tmp_path)
    events = [  # 乱序写入，验证排序+索引链路
        {"ts_ms": BASE + 200, "switch": "sw1", "port": "Eth1/1", "event": "link-up"},
        {"ts_ms": BASE + 100, "switch": "sw1", "port": "Eth1/1", "event": "link-down"},
        {"ts_ms": BASE + 150, "switch": "sw1", "port": "Eth1/2", "event": "link-up"},
        {"ts_ms": BASE + 120, "switch": "sw2", "port": "Eth1/1", "event": "link-down"},
        {"ts_ms": BASE + 130, "switch": "sw1", "port": "Eth1/1", "event": "flap"},  # 不改变状态
    ]
    write_log(settings.logs_dir, "a.ndjson", events)
    with TestClient(create_app(settings)) as client:
        job_id = run_job(client, "a.ndjson")
        # 完整覆盖 [起,止)：未知 → 掉线 → 在线
        assert query_intervals(client, job_id, "sw1", "Eth1/1", BASE, BASE + 300) == [
            iv(BASE, BASE + 100, "unknown"),
            iv(BASE + 100, BASE + 200, "down"),
            iv(BASE + 200, BASE + 300, "online"),
        ]
        # 起点落在掉线区间内：起点前事件纳入，状态被正确带入
        assert query_intervals(client, job_id, "sw1", "Eth1/1",
                               BASE + 150, BASE + 250) == [
            iv(BASE + 150, BASE + 200, "down"),
            iv(BASE + 200, BASE + 250, "online"),
        ]
        # 起点在最后一个事件之后：整段继承起点前状态
        assert query_intervals(client, job_id, "sw1", "Eth1/1",
                               BASE + 250, BASE + 300) == [
            iv(BASE + 250, BASE + 300, "online"),
        ]
        # 其它端口 / 其它交换机同名端口：互不影响
        assert query_intervals(client, job_id, "sw1", "Eth1/2", BASE, BASE + 300) == [
            iv(BASE, BASE + 150, "unknown"),
            iv(BASE + 150, BASE + 300, "online"),
        ]
        assert query_intervals(client, job_id, "sw2", "Eth1/1", BASE, BASE + 300) == [
            iv(BASE, BASE + 120, "unknown"),
            iv(BASE + 120, BASE + 300, "down"),
        ]
        # 从未上报的端口：整段未知
        assert query_intervals(client, job_id, "sw1", "Eth1/9", BASE, BASE + 300) == [
            iv(BASE, BASE + 300, "unknown"),
        ]


def test_reboot_resets_switch_ports_only(tmp_path):
    """设备重启：本机各端口回到未知，其它交换机不受影响。"""
    settings = make_settings(tmp_path)
    events = [
        {"ts_ms": BASE + 100, "switch": "sw1", "port": "Eth1/1", "event": "link-up"},
        {"ts_ms": BASE + 100, "switch": "sw1", "port": "Eth1/2", "event": "link-down"},
        {"ts_ms": BASE + 100, "switch": "sw2", "port": "Eth1/1", "event": "link-up"},
        {"ts_ms": BASE + 200, "switch": "sw1", "event": "reboot"},  # 无需 port
        {"ts_ms": BASE + 300, "switch": "sw1", "port": "Eth1/1", "event": "link-down"},
    ]
    write_log(settings.logs_dir, "r.ndjson", events)
    with TestClient(create_app(settings)) as client:
        job_id = run_job(client, "r.ndjson")
        # 重启后回到未知，直到下一次上报
        assert query_intervals(client, job_id, "sw1", "Eth1/1", BASE, BASE + 400) == [
            iv(BASE, BASE + 100, "unknown"),
            iv(BASE + 100, BASE + 200, "online"),
            iv(BASE + 200, BASE + 300, "unknown"),
            iv(BASE + 300, BASE + 400, "down"),
        ]
        assert query_intervals(client, job_id, "sw1", "Eth1/2", BASE, BASE + 400) == [
            iv(BASE, BASE + 100, "unknown"),
            iv(BASE + 100, BASE + 200, "down"),
            iv(BASE + 200, BASE + 400, "unknown"),
        ]
        # 其它交换机不受 sw1 重启影响
        assert query_intervals(client, job_id, "sw2", "Eth1/1", BASE, BASE + 400) == [
            iv(BASE, BASE + 100, "unknown"),
            iv(BASE + 100, BASE + 400, "online"),
        ]
        # 起点在重启之后：起点前的 reboot 也被纳入（状态为未知）
        assert query_intervals(client, job_id, "sw1", "Eth1/1",
                               BASE + 250, BASE + 350) == [
            iv(BASE + 250, BASE + 300, "unknown"),
            iv(BASE + 300, BASE + 350, "down"),
        ]


def test_same_ms_events_apply_in_result_line_order(tmp_path):
    """同一毫秒的事件按成品行序（稳定排序后的原始行序）生效。"""
    settings = make_settings(tmp_path)
    events = [
        # sw1/Eth1/1：同一毫秒先 up 后 down → 最终掉线
        {"ts_ms": BASE + 100, "switch": "sw1", "port": "Eth1/1", "event": "link-up"},
        {"ts_ms": BASE + 100, "switch": "sw1", "port": "Eth1/1", "event": "link-down"},
        # sw1/Eth1/2：同一毫秒先 down 后 up → 最终在线
        {"ts_ms": BASE + 100, "switch": "sw1", "port": "Eth1/2", "event": "link-down"},
        {"ts_ms": BASE + 100, "switch": "sw1", "port": "Eth1/2", "event": "link-up"},
        # sw2：同一毫秒先 reboot 后 link-up → 上报在后，最终在线
        {"ts_ms": BASE + 100, "switch": "sw2", "event": "reboot"},
        {"ts_ms": BASE + 100, "switch": "sw2", "port": "Eth1/1", "event": "link-up"},
        # sw3：同一毫秒先 link-up 后 reboot → 重启在后，回到未知
        {"ts_ms": BASE + 100, "switch": "sw3", "port": "Eth1/1", "event": "link-up"},
        {"ts_ms": BASE + 100, "switch": "sw3", "event": "reboot"},
    ]
    write_log(settings.logs_dir, "tie.ndjson", events)
    with TestClient(create_app(settings)) as client:
        job_id = run_job(client, "tie.ndjson")
        assert query_intervals(client, job_id, "sw1", "Eth1/1", BASE, BASE + 200) == [
            iv(BASE, BASE + 100, "unknown"),
            iv(BASE + 100, BASE + 200, "down"),
        ]
        assert query_intervals(client, job_id, "sw1", "Eth1/2", BASE, BASE + 200) == [
            iv(BASE, BASE + 100, "unknown"),
            iv(BASE + 100, BASE + 200, "online"),
        ]
        assert query_intervals(client, job_id, "sw2", "Eth1/1", BASE, BASE + 200) == [
            iv(BASE, BASE + 100, "unknown"),
            iv(BASE + 100, BASE + 200, "online"),
        ]
        assert query_intervals(client, job_id, "sw3", "Eth1/1", BASE, BASE + 200) == [
            iv(BASE, BASE + 200, "unknown"),
        ]


# ---------- 索引构建生命周期 ----------

def _gate_build(monkeypatch):
    """把 build_index 换成受控版本，返回 (entered, release, calls)。"""
    real_build = portindex_module.build_index
    entered = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def gated(path, ts_field):
        calls["n"] += 1
        entered.set()
        assert release.wait(timeout=15), "测试未及时放行索引构建"
        return real_build(path, ts_field)

    monkeypatch.setattr(portindex_module, "build_index", gated)
    return entered, release, calls


def test_building_reports_status_and_download_stays_available(tmp_path, monkeypatch):
    """索引完整前只报告构建状态；构建期间原成品仍可下载。"""
    settings = make_settings(tmp_path)
    events = [
        {"ts_ms": BASE + 100, "switch": "sw1", "port": "Eth1/1", "event": "link-up"},
    ]
    write_log(settings.logs_dir, "g.ndjson", events)
    entered, release, calls = _gate_build(monkeypatch)
    with TestClient(create_app(settings)) as client:
        job_id = run_job(client, "g.ndjson")

        r = client.get(f"/api/jobs/{job_id}/port-status",
                       params={"switch": "sw1", "port": "Eth1/1",
                               "start_ms": BASE, "end_ms": BASE + 200})
        assert r.status_code == 202  # 首次查询触发构建，只报告状态
        assert r.json()["index_state"] == "building"
        assert r.json()["attempts"] == 1
        assert entered.wait(timeout=10)

        r = client.get(f"/api/jobs/{job_id}/port-status",
                       params={"switch": "sw1", "port": "Eth1/1",
                               "start_ms": BASE, "end_ms": BASE + 200})
        assert r.status_code == 202  # 构建未完，仍只报告状态
        assert calls["n"] == 1       # 并发/重复查询共用同一次构建

        d = client.get(f"/api/jobs/{job_id}/download")
        assert d.status_code == 200  # 构建期间原成品仍可下载
        assert "link-up" in d.text

        release.set()
        assert query_intervals(client, job_id, "sw1", "Eth1/1", BASE, BASE + 200) == [
            iv(BASE, BASE + 100, "unknown"),
            iv(BASE + 100, BASE + 200, "online"),
        ]
        assert calls["n"] == 1  # 就绪后查询不再触发构建


def test_build_failure_reports_reason_retries_and_download_survives(tmp_path, monkeypatch):
    """索引构建失败：说明原因、可重试、失败后原成品仍可下载。"""
    settings = make_settings(tmp_path)
    events = [
        {"ts_ms": BASE + 100, "switch": "sw1", "port": "Eth1/1", "event": "link-down"},
    ]
    write_log(settings.logs_dir, "f.ndjson", events)
    real_build = portindex_module.build_index
    attempts = {"n": 0}

    def flaky(path, ts_field):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("simulated disk hiccup")
        return real_build(path, ts_field)

    monkeypatch.setattr(portindex_module, "build_index", flaky)
    params = {"switch": "sw1", "port": "Eth1/1", "start_ms": BASE, "end_ms": BASE + 200}
    with TestClient(create_app(settings)) as client:
        job_id = run_job(client, "f.ndjson")

        # 第一次构建失败 → 409 + 原因
        deadline = time.monotonic() + 10
        detail = None
        while time.monotonic() < deadline:
            r = client.get(f"/api/jobs/{job_id}/port-status", params=params)
            if r.status_code == 409:
                detail = r.json()["detail"]
                break
            assert r.status_code == 202, r.text
            time.sleep(0.02)
        assert detail is not None, "未等到失败状态"
        assert "simulated disk hiccup" in detail  # 说明原因
        assert "重试" in detail

        d = client.get(f"/api/jobs/{job_id}/download")
        assert d.status_code == 200  # 失败后原成品仍可下载

        # 重新查询即重试 → 成功
        assert query_intervals(client, job_id, "sw1", "Eth1/1", BASE, BASE + 200) == [
            iv(BASE, BASE + 100, "unknown"),
            iv(BASE + 100, BASE + 200, "down"),
        ]
        assert attempts["n"] == 2


def test_build_failure_from_corrupt_result_then_retry_after_fix(tmp_path):
    """成品被写坏 → 构建失败带行号；修复后重试成功；期间下载不受影响。"""
    settings = make_settings(tmp_path)
    events = [
        {"ts_ms": BASE + 100, "switch": "sw1", "port": "Eth1/1", "event": "link-up"},
        {"ts_ms": BASE + 200, "switch": "sw1", "port": "Eth1/1", "event": "link-down"},
    ]
    write_log(settings.logs_dir, "c.ndjson", events)
    params = {"switch": "sw1", "port": "Eth1/1", "start_ms": BASE, "end_ms": BASE + 300}
    with TestClient(create_app(settings)) as client:
        job_id = run_job(client, "c.ndjson")
        result_file = settings.results_dir / f"{job_id}.ndjson"
        original = result_file.read_bytes()

        with result_file.open("ab") as fh:  # 追加非法行，破坏成品
            fh.write(b"this-is-not-json\n")

        deadline = time.monotonic() + 10
        detail = None
        while time.monotonic() < deadline:
            r = client.get(f"/api/jobs/{job_id}/port-status", params=params)
            if r.status_code == 409:
                detail = r.json()["detail"]
                break
            assert r.status_code == 202, r.text
            time.sleep(0.02)
        assert detail is not None, "未等到失败状态"
        assert "第 3 行" in detail  # 失败原因含行号

        d = client.get(f"/api/jobs/{job_id}/download")
        assert d.status_code == 200  # 失败后成品仍可下载（内容原样）

        result_file.write_bytes(original)  # 修复成品
        assert query_intervals(client, job_id, "sw1", "Eth1/1", BASE, BASE + 300) == [
            iv(BASE, BASE + 100, "unknown"),
            iv(BASE + 100, BASE + 200, "online"),
            iv(BASE + 200, BASE + 300, "down"),
        ]


def test_concurrent_queries_share_single_build(tmp_path, monkeypatch):
    """同一作业的并发查询共用同一次索引构建。"""
    settings = make_settings(tmp_path)
    events = [
        {"ts_ms": BASE + 10, "switch": "sw1", "port": "Eth1/1", "event": "link-down"},
        {"ts_ms": BASE + 20, "switch": "sw1", "port": "Eth1/1", "event": "link-up"},
    ] + [  # 其它端口的填充事件，让构建有点工作量
        {"ts_ms": BASE + i, "switch": "sw1", "port": f"Eth1/{i % 3 + 2}",
         "event": "link-up" if i % 2 else "link-down"}
        for i in range(40)
    ]
    write_log(settings.logs_dir, "cc.ndjson", events)
    entered, release, calls = _gate_build(monkeypatch)
    with TestClient(create_app(settings)) as client:
        job_id = run_job(client, "cc.ndjson")
        manager = client.app.state.manager

        first = manager.port_status(job_id, "sw1", "Eth1/1", BASE, BASE + 100)
        assert first.state.value == "building"
        assert entered.wait(timeout=10)

        results, errors = [], []

        def worker():
            try:
                results.append(manager.port_status(job_id, "sw1", "Eth1/1", BASE, BASE + 100))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors
        assert all(r.state.value == "building" for r in results)  # 并发查询只见到构建中
        release.set()

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            done = manager.port_status(job_id, "sw1", "Eth1/1", BASE, BASE + 100)
            if done.state.value == "ready":
                break
            time.sleep(0.02)
        assert done.state.value == "ready"
        assert done.intervals == [
            iv(BASE, BASE + 10, "unknown"),
            iv(BASE + 10, BASE + 20, "down"),
            iv(BASE + 20, BASE + 100, "online"),
        ]
        assert calls["n"] == 1  # 全程只构建了一次


def test_ready_index_serves_queries_without_rescanning_result(tmp_path):
    """索引就绪后查询不再重扫成品：成品被改写，查询结果仍来自索引。"""
    settings = make_settings(tmp_path)
    events = [
        {"ts_ms": BASE + 100, "switch": "sw1", "port": "Eth1/1", "event": "link-up"},
    ]
    write_log(settings.logs_dir, "n.ndjson", events)
    with TestClient(create_app(settings)) as client:
        job_id = run_job(client, "n.ndjson")
        before = query_intervals(client, job_id, "sw1", "Eth1/1", BASE, BASE + 200)
        assert before == [iv(BASE, BASE + 100, "unknown"), iv(BASE + 100, BASE + 200, "online")]

        # 改写成品：若查询重扫文件，结果必然改变
        result_file = settings.results_dir / f"{job_id}.ndjson"
        result_file.write_text(
            json.dumps({"ts_ms": BASE + 50, "switch": "sw1", "event": "reboot"}) + "\n",
            encoding="utf-8",
        )
        assert query_intervals(client, job_id, "sw1", "Eth1/1", BASE, BASE + 200) == before


# ---------- 查询门禁与参数校验 ----------

def test_query_guards(tmp_path, monkeypatch):
    """未完成作业 409、不存在 404、起止非法 400、参数缺失 422。"""
    settings = make_settings(tmp_path)
    write_log(settings.logs_dir, "ok.ndjson",
              [{"ts_ms": BASE, "switch": "sw1", "port": "Eth1/1", "event": "link-up"}])
    (settings.logs_dir / "bad.ndjson").write_text('{"ts_ms": "oops"}\n', encoding="utf-8")

    real_sort = jobs_module.external_sort
    entered = threading.Event()
    release = threading.Event()

    def gated_sort(*args, **kwargs):
        entered.set()
        release.wait(timeout=15)
        return real_sort(*args, **kwargs)

    monkeypatch.setattr(jobs_module, "external_sort", gated_sort)
    params = {"switch": "sw1", "port": "Eth1/1", "start_ms": BASE, "end_ms": BASE + 100}
    with TestClient(create_app(settings)) as client:
        r = client.post("/api/jobs", json={"filename": "ok.ndjson"})
        running_id = r.json()["job_id"]
        assert entered.wait(timeout=10)
        r = client.get(f"/api/jobs/{running_id}/port-status", params=params)
        assert r.status_code == 409  # 运行中不可查询
        assert "仅已完成作业" in r.json()["detail"]
        release.set()
        job_id = run_job(client, "ok.ndjson")

        bad_id = client.post("/api/jobs", json={"filename": "bad.ndjson"}).json()["job_id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if client.get(f"/api/jobs/{bad_id}").json()["state"] == "failed":
                break
            time.sleep(0.02)
        r = client.get(f"/api/jobs/{bad_id}/port-status", params=params)
        assert r.status_code == 409  # 失败作业不可查询

        assert client.get("/api/jobs/nope/port-status", params=params).status_code == 404

        bad_range = dict(params, start_ms=BASE + 100, end_ms=BASE + 100)
        r = client.get(f"/api/jobs/{job_id}/port-status", params=bad_range)
        assert r.status_code == 400  # 起 == 止
        bad_range = dict(params, start_ms=BASE + 200, end_ms=BASE + 100)
        assert client.get(f"/api/jobs/{job_id}/port-status", params=bad_range).status_code == 400

        r = client.get(f"/api/jobs/{job_id}/port-status",
                       params={"switch": "  ", "port": "Eth1/1",
                               "start_ms": BASE, "end_ms": BASE + 100})
        assert r.status_code == 400  # 空 switch
        r = client.get(f"/api/jobs/{job_id}/port-status",
                       params={"switch": "sw1", "start_ms": BASE, "end_ms": BASE + 100})
        assert r.status_code == 422  # 缺 port


# ---------- 索引单元测试 ----------

def test_port_index_intervals_unit():
    """手工索引：边界（事件恰在起/止）、reboot 决胜、相邻同态合并。"""
    index = PortIndex(
        ports={
            ("sw1", "Eth1/1"): [(100, 1, "online"), (200, 4, "down")],
            ("sw1", "Eth1/2"): [(150, 2, "online")],
        },
        reboots={"sw1": [(180, 3)]},
        event_count=4,
    )
    assert index.intervals("sw1", "Eth1/1", 0, 300) == [
        iv(0, 100, "unknown"), iv(100, 180, "online"),
        iv(180, 200, "unknown"), iv(200, 300, "down"),
    ]
    assert index.intervals("sw1", "Eth1/1", 100, 200) == [  # 事件恰在起点生效
        iv(100, 180, "online"), iv(180, 200, "unknown"),
    ]
    assert index.intervals("sw1", "Eth1/1", 0, 100) == [iv(0, 100, "unknown")]  # 终点事件不算
    assert index.intervals("sw1", "Eth1/1", 200, 250) == [iv(200, 250, "down")]
    assert index.intervals("sw1", "Eth1/2", 0, 300) == [  # reboot 后整段未知
        iv(0, 150, "unknown"), iv(150, 180, "online"), iv(180, 300, "unknown"),
    ]
    assert index.intervals("sw9", "Eth9/9", 0, 300) == [iv(0, 300, "unknown")]


def test_build_index_unit_skips_irrelevant_and_stays_linear(tmp_path):
    """逐行构建：无关事件不入索引；索引条数 ≤ 有效行数；同毫秒按行序。"""
    path = tmp_path / "r.ndjson"
    lines = [
        {"ts_ms": 100, "switch": "sw1", "port": "Eth1/1", "event": "link-up"},
        {"ts_ms": 100, "switch": "sw1", "event": "reboot"},  # 同毫秒，行序在后 → 覆盖 up
        {"ts_ms": 200, "switch": "sw1", "port": "Eth1/1", "event": "link-down"},
        {"ts_ms": 300, "switch": "sw1", "port": "Eth1/1", "event": "flap"},       # 忽略
        {"ts_ms": 400, "switch": "sw1", "port": "Eth1/1", "event": "speed-change"},  # 忽略
        {"ts_ms": 500, "switch": "sw2", "event": "reboot"},
        {"ts_ms": 600, "switch": "sw2", "port": "Eth1/9", "event": "link-up"},
        {"ts_ms": 700, "switch": "sw2", "port": "Eth1/9"},  # 无 event 字段，忽略
        {"ts_ms": 800, "port": "Eth1/1", "event": "link-up"},  # 缺 switch，忽略
        {"ts_ms": 900, "switch": "sw2", "event": "link-down"},  # 缺 port，忽略
    ]
    path.write_text("".join(json.dumps(e) + "\n" for e in lines), encoding="utf-8")
    index = build_index(path, "ts_ms")

    entries = sum(len(v) for v in index._ports.values()) + \
        sum(len(v) for v in index._reboots.values())
    assert entries == 5 <= index.event_count == 10  # 空间随日志条数线性（只少不多）

    assert index.intervals("sw1", "Eth1/1", 0, 1000) == [
        iv(0, 200, "unknown"),   # 同毫秒 up 被其后的 reboot 覆盖
        iv(200, 1000, "down"),
    ]
    assert index.intervals("sw2", "Eth1/9", 0, 1000) == [
        iv(0, 600, "unknown"),
        iv(600, 1000, "online"),
    ]


def test_build_index_unit_failures(tmp_path):
    """坏 JSON 带行号、缺时间戳字段、时间戳回退、文件缺失：都说明原因。"""
    bad_json = tmp_path / "bad.ndjson"
    bad_json.write_text('{"ts_ms": 1, "event": "reboot", "switch": "s"}\nnot-json\n',
                        encoding="utf-8")
    with pytest.raises(SortError, match="第 2 行"):
        build_index(bad_json, "ts_ms")

    no_ts = tmp_path / "no_ts.ndjson"
    no_ts.write_text('{"switch": "s", "event": "reboot"}\n', encoding="utf-8")
    with pytest.raises(SortError, match="缺少时间戳字段"):
        build_index(no_ts, "ts_ms")

    regress = tmp_path / "regress.ndjson"
    regress.write_text(
        '{"ts_ms": 200, "switch": "s", "event": "reboot"}\n'
        '{"ts_ms": 100, "switch": "s", "event": "reboot"}\n',
        encoding="utf-8",
    )
    with pytest.raises(SortError, match="时间戳回退"):
        build_index(regress, "ts_ms")

    with pytest.raises(SortError, match="读取结果文件失败"):
        build_index(tmp_path / "missing.ndjson", "ts_ms")


def test_index_manager_retry_only_after_failure(tmp_path):
    """IndexManager：失败可重试；就绪后不再重建。"""
    good = tmp_path / "good.ndjson"
    good.write_text(
        '{"ts_ms": 1, "switch": "s", "port": "p", "event": "link-up"}\n', encoding="utf-8"
    )
    manager = IndexManager(result_path=lambda job_id: good, ts_field="ts_ms")
    try:
        r = manager.query("j1", "s", "p", 0, 10)
        deadline = time.monotonic() + 10
        while r.state.value != "ready" and time.monotonic() < deadline:
            time.sleep(0.02)
            r = manager.query("j1", "s", "p", 0, 10)
        assert r.intervals == [iv(0, 1, "unknown"), iv(1, 10, "online")]
        attempts = r.attempts
        r2 = manager.query("j1", "s", "p", 0, 10)
        assert r2.attempts == attempts  # 就绪后复用索引，不再重建
    finally:
        manager.shutdown()
