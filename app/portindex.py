"""端口状态区间索引：从已完成作业的排序成品构建，回答 [起,止) 的在线/掉线/未知区间。

状态语义：
- 端口初始为 unknown；link-up → online，link-down → down；
- reboot（无需 port 字段）使该交换机全部端口回到 unknown；
- 同一毫秒的事件按成品文件行序生效（seq = 成品行号；成品已按 (ts_ms, 原始行序) 稳定排序，
  逐行读取时 seq 递增，恰好就是同毫秒事件的生效顺序）；
- 查询 [start_ms, end_ms)：起点状态由 ≤ start_ms 的最近一条事件决定，
  起点前的历史由此自然纳入，无需逐条回放。

索引结构（每行日志至多产生一条索引项，空间随日志条数线性增长）：
- ports:   {(switch, port): [(ts_ms, seq, state), ...]}  仅 link-up/link-down 行
- reboots: {switch: [(ts_ms, seq), ...]}                 仅 reboot 行
其它事件类型（flap/speed-change/...）不改变链路状态，不进入索引。

查询复杂度：二分定位起点状态 O(log n)，随后只归并该端口事件与该机 reboot 中
落在 (start_ms, end_ms) 内的部分——不重扫成品文件、不遍历其它端口的事件、
不遍历所选端口在起点前的全部历史。

构建生命周期（每个已完成作业一份，进程内存中）：
- 首次查询触发构建；同一作业的并发查询共用同一次构建（single-flight）；
- 索引完整前查询只报告构建状态（building）；
- 失败向观察到它的查询报告原因，并随即触发重新构建（重试），
  后续查询先见到 building、构建成功后见到区间结果；
- 构建只读成品文件，构建期间与失败后成品照常可下载。
"""
from __future__ import annotations

import json
import math
import threading
import time
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from .sorter import SortError, parse_ts_ms

ONLINE = "online"
DOWN = "down"
UNKNOWN = "unknown"

_LINK_STATES = {"link-up": ONLINE, "link-down": DOWN}
_REBOOT = "reboot"


class IndexState(str, Enum):
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


@dataclass
class QueryResult:
    """一次端口状态查询的结果：要么给出区间，要么报告索引构建状态。"""

    state: IndexState
    intervals: Optional[list[dict]] = None
    error: Optional[str] = None
    attempts: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class PortIndex:
    """不可变索引：构建完成后只读，区间计算无副作用，可被并发查询共享。"""

    def __init__(self, ports: dict, reboots: dict, event_count: int):
        self._ports = ports      # (switch, port) -> [(ts_ms, seq, state)]，按 (ts, seq) 有序
        self._reboots = reboots  # switch -> [(ts_ms, seq)]，按 (ts, seq) 有序
        self.event_count = event_count  # 成品有效行数（索引条数 ≤ 该值）

    def intervals(self, switch: str, port: str, start_ms: int, end_ms: int) -> list[dict]:
        """返回覆盖 [start_ms, end_ms) 的极大等状态区间列表（要求 start_ms < end_ms）。"""
        events = self._ports.get((switch, port), ())
        reboots = self._reboots.get(switch, ())

        # 起点状态：≤ start_ms 的最近一条端口事件或 reboot，按 (ts_ms, seq) 决胜。
        # 每条事件都是绝对赋值（online/down/unknown），所以最近一条即当前状态。
        i = bisect_right(events, (start_ms, math.inf, "")) - 1
        j = bisect_right(reboots, (start_ms, math.inf)) - 1
        state = UNKNOWN
        winner = None  # 胜出的 (ts_ms, seq)
        if i >= 0:
            state = events[i][2]
            winner = (events[i][0], events[i][1])
        if j >= 0 and (winner is None or (reboots[j][0], reboots[j][1]) > winner):
            state = UNKNOWN

        # (start_ms, end_ms) 内的事件：两路有序序列归并，只触及落在区间内的条目
        i = bisect_right(events, (start_ms, math.inf, ""))
        ie = bisect_left(events, (end_ms,))
        j = bisect_right(reboots, (start_ms, math.inf))
        je = bisect_left(reboots, (end_ms,))

        out: list[dict] = []
        cur = start_ms
        while i < ie or j < je:
            if j >= je or (i < ie and (events[i][0], events[i][1]) < (reboots[j][0], reboots[j][1])):
                ts, _, nxt = events[i]
                i += 1
            else:
                ts, _ = reboots[j]
                j += 1
                nxt = UNKNOWN
            if nxt == state:
                continue  # 状态不变，不产生边界
            if ts > cur:
                out.append({"start_ms": cur, "end_ms": ts, "state": state})
                cur = ts
            # ts == cur：同一毫秒的后续事件按行序继续生效，只更新状态
            state = nxt
        out.append({"start_ms": cur, "end_ms": end_ms, "state": state})

        # 同毫秒内状态 A→B→A 会留下相邻同态区间，合并为极大区间
        merged: list[dict] = []
        for iv in out:
            if merged and merged[-1]["state"] == iv["state"]:
                merged[-1]["end_ms"] = iv["end_ms"]
            else:
                merged.append(iv)
        return merged


def _line_ts(obj: dict, lineno: int, ts_field: str) -> int:
    if ts_field not in obj:
        raise SortError(f"第 {lineno} 行：缺少时间戳字段 {ts_field!r}")
    return parse_ts_ms(obj[ts_field], lineno, ts_field)


def build_index(path: Path, ts_field: str) -> PortIndex:
    """逐行流式读取排序成品，构建端口状态索引。

    失败抛出带行号/原因的 SortError（JSON 非法、时间戳缺失或非法、
    时间戳回退——成品不是有效排序结果、读写错误）。只读文件，不做任何修改。
    """
    ports: dict[tuple[str, str], list] = {}
    reboots: dict[str, list] = {}
    seq = 0       # 成品行序（跳过空白行）：同毫秒事件按此生效
    last_ts = None
    try:
        fin = open(path, "rb")
    except OSError as exc:
        raise SortError(f"读取结果文件失败：{exc}（{path}）") from exc
    with fin:
        while True:
            try:
                raw = fin.readline()
            except OSError as exc:
                raise SortError(f"读取结果文件失败（第 {seq + 1} 行附近）：{exc}") from exc
            if not raw:
                break
            if not raw.strip():
                continue  # 空白行不参与行序
            seq += 1
            try:
                obj = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SortError(f"第 {seq} 行：不是合法 JSON（{exc}）") from exc
            if not isinstance(obj, dict):
                raise SortError(f"第 {seq} 行：JSON 不是对象")
            event = obj.get("event")
            if event in _LINK_STATES:
                switch = obj.get("switch")
                port = obj.get("port")
                if not switch or not port:
                    continue  # 无法归属到端口，不参与状态推断
                ts = _line_ts(obj, seq, ts_field)
                if last_ts is not None and ts < last_ts:
                    raise SortError(
                        f"第 {seq} 行：时间戳回退（{ts} < {last_ts}），成品不是有效排序结果"
                    )
                last_ts = ts
                entries = ports.setdefault((switch, port), [])
                entries.append((ts, seq, _LINK_STATES[event]))
            elif event == _REBOOT:
                switch = obj.get("switch")
                if not switch:
                    continue  # reboot 无需 port，但必须能归属到交换机
                ts = _line_ts(obj, seq, ts_field)
                if last_ts is not None and ts < last_ts:
                    raise SortError(
                        f"第 {seq} 行：时间戳回退（{ts} < {last_ts}），成品不是有效排序结果"
                    )
                last_ts = ts
                entries = reboots.setdefault(switch, [])
                entries.append((ts, seq))
            # 其它事件（flap/speed-change/duplex-mismatch/...）不改变链路状态，不入索引
    return PortIndex(ports, reboots, seq)


@dataclass
class _Entry:
    state: IndexState
    index: Optional[PortIndex] = None
    error: Optional[str] = None
    attempts: int = 0
    started_at: float = 0.0
    finished_at: Optional[float] = None


class IndexManager:
    """按作业构建/缓存端口状态索引（进程内存，单实例）。

    - 首次查询或失败后重试时触发构建，构建在独立线程池中进行；
    - 同一作业同一时刻至多一次构建，并发查询共用该次构建；
    - 索引就绪后查询不再触碰成品文件。
    """

    def __init__(self, result_path: Callable[[str], Path], ts_field: str, workers: int = 2):
        self._result_path = result_path
        self._ts_field = ts_field
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="index-builder"
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def query(self, job_id: str, switch: str, port: str,
              start_ms: int, end_ms: int) -> QueryResult:
        """查询端口状态区间；索引未就绪时只报告构建状态。

        首次查询触发构建；观察到 FAILED 的查询返回失败原因（FAILED），
        并同时触发重新构建——重试由此自然发生，后续查询先见到 building 再见到结果。
        """
        with self._lock:
            entry = self._entries.get(job_id)
            if entry is None:
                entry = self._start_build_locked(job_id, attempts=1)
                return self._status_of(entry)
            if entry.state == IndexState.FAILED:
                error, attempts = entry.error, entry.attempts
                self._start_build_locked(job_id, attempts=attempts + 1)
                return QueryResult(state=IndexState.FAILED, error=error, attempts=attempts)
            if entry.state == IndexState.BUILDING:
                return self._status_of(entry)
            index = entry.index
            attempts = entry.attempts
        # 锁外计算：READY 后条目不再变更，索引不可变
        return QueryResult(
            state=IndexState.READY,
            intervals=index.intervals(switch, port, start_ms, end_ms),
            attempts=attempts,
        )

    def _status_of(self, entry: _Entry) -> QueryResult:
        return QueryResult(
            state=entry.state,
            error=entry.error,
            attempts=entry.attempts,
            started_at=entry.started_at,
            finished_at=entry.finished_at,
        )

    def _start_build_locked(self, job_id: str, attempts: int) -> _Entry:
        """登记一个 building 条目并提交构建任务；调用须持锁。并发查询共用该条目。"""
        entry = _Entry(state=IndexState.BUILDING, attempts=attempts,
                       started_at=time.time())
        self._entries[job_id] = entry
        self._executor.submit(self._build, job_id, entry)
        return entry

    def _build(self, job_id: str, entry: _Entry) -> None:
        try:
            index = build_index(self._result_path(job_id), self._ts_field)
            error = None
        except Exception as exc:  # 失败必须说明原因
            index = None
            error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            if self._entries.get(job_id) is not entry:
                return  # 条目已被替换（防御；正常不会发生）
            if error is None:
                entry.state = IndexState.READY
                entry.index = index
            else:
                entry.state = IndexState.FAILED
                entry.error = error
            entry.finished_at = time.time()
