import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any

from core.db import get_conn, get_step_outcomes

_log = logging.getLogger("jarvis.task_queue")


class TaskStatus(Enum):
    PENDING     = "pending"
    RUNNING     = "running"
    COMPLETED   = "completed"
    FAILED      = "failed"
    CANCELLED   = "cancelled"
    INTERRUPTED = "interrupted"   # DB-only: orphaned pending/running row found at startup, never resumed


class TaskPriority(Enum):
    LOW    = 3
    NORMAL = 2
    HIGH   = 1


@dataclass(order=True)
class Task:
    priority:    int
    created_at:  float = field(compare=False)
    task_id:     str   = field(compare=False)
    goal:        str   = field(compare=False)
    status:      TaskStatus = field(compare=False, default=TaskStatus.PENDING)
    result:      Any        = field(compare=False, default=None)
    error:       str        = field(compare=False, default="")
    speak:       Any        = field(compare=False, default=None)
    on_complete: Any        = field(compare=False, default=None)
    cancel_flag: threading.Event = field(compare=False, default_factory=threading.Event)
    # Whether a live interactive session submitted this task (vs. some
    # future system/scheduled trigger). Threaded through to policy
    # evaluation for the task's steps (core/tool_gate.dispatch_tool) —
    # doesn't change *whether* an ask-and-wait step pauses (player is None
    # either way on this thread, see agent/executor.py), only the context
    # noted on the resulting approval prompt.
    submitted_interactively: bool = field(compare=False, default=True)
    # Headless-extraction phase, Step 1.3: who submitted this task — one
    # of core/policy.py's CALLER_CLASSES. Defaults to 'desktop' so every
    # pre-existing submit() call site is unaffected. Threaded through to
    # AgentExecutor.execute() so every task_events row it logs for this
    # task's steps is correctly attributed (see agent/executor.py).
    caller_class: str = field(compare=False, default="desktop")


def _reconcile_interrupted_tasks() -> None:
    """Called once at startup (see TaskQueue.start()): AgentExecutor isn't
    step-checkpointed, so a task still 'pending'/'running' in the DB from
    a previous process lifetime can't be safely resumed — mark it
    'interrupted' instead of silently losing it or attempting to continue
    mid-flight."""
    conn = get_conn()
    with conn:
        cur = conn.execute(
            "UPDATE tasks SET status = ?, updated_at = ? WHERE status IN (?, ?)",
            (TaskStatus.INTERRUPTED.value, time.time(), TaskStatus.PENDING.value, TaskStatus.RUNNING.value),
        )
        n = cur.rowcount
    if n:
        _log.warning("Marked %d orphaned task(s) from a previous run as interrupted.", n)


def _reconcile_orphaned_approvals() -> None:
    """Called once at startup, in the same reconciliation pass as
    _reconcile_interrupted_tasks (see TaskQueue.start()) — same rationale,
    different table: a background ask-and-wait approval
    (core/task_approval.py's TASK_APPROVAL) still 'pending' from a
    previous process lifetime has no live worker thread left to resume
    it. TASK_APPROVAL's pending-approval registry is in-memory only and
    doesn't survive a restart, so nothing will ever call answer() on it —
    and by the time this runs (start of TaskQueue.start(), before any
    worker thread exists), that's true of every 'pending' row without
    needing to cross-reference which task it belongs to.

    Marked 'expired', not reused as 'timeout' — 'timeout' already means
    something more specific in this vocabulary (an in-process wait that
    elapsed while the app was still live and could have been answered);
    collapsing the two would make that distinction unrecoverable in the
    audit trail. See core/db.py's approvals table for the full outcome
    vocabulary this stays consistent with (approved | denied | timeout |
    expired | pending | no_player)."""
    conn = get_conn()
    with conn:
        cur = conn.execute(
            "UPDATE approvals SET answered_at = ?, outcome = 'expired' WHERE outcome = 'pending'",
            (time.time(),),
        )
        n = cur.rowcount
    if n:
        _log.warning("Marked %d orphaned approval(s) from a previous run as expired.", n)


def _last_terminal_failure_detail(task_id: str) -> str | None:
    """Headless-extraction phase, Step 1.6c. AgentExecutor.execute()
    never raises for "ran out of replan attempts" / "no valid plan" / a
    denied or hard-denied step it ultimately gave up on — it returns a
    plain string, same convention as every tool result in this codebase
    (see agent/executor.py). Task.status must not blindly read that as
    success: this checks the real, durable outcome via
    get_step_outcomes() — the same helper _summarize() itself trusts —
    rather than pattern-matching the returned message text.

    Returns the last step segment's own detail string (e.g.
    "approval_denied: ..." or "hard_denied: ...", the exact vocabulary
    _short_circuit_reason() already established) if the task's most
    recently logged segment ended in 'failed' — meaning the task
    genuinely gave up on that failure rather than recovering from it via
    a later replan attempt. A step that failed but was later followed by
    a real 'done'/'skipped' segment (a successful replan-recovery) is
    NOT reported here — get_step_outcomes()'s segments are chronological,
    so the *last* one is what execute() actually ended on.

    None if the real last outcome was 'done'/'skipped' (a genuine
    success or acceptable skip) or if there are no outcomes logged at
    all — the "no valid plan could ever be created, not even a first
    attempt" case has no task_events history to check against and is
    deliberately out of this fix's scope; it keeps its existing
    'completed' behavior."""
    outcomes = get_step_outcomes(task_id)
    if not outcomes:
        return None
    last = outcomes[-1]
    if last["status"] == "failed":
        return last["detail"] or "failed"
    return None


class TaskQueue:
    def __init__(self, max_concurrent: int = 1):
        self._queue:        list[Task]       = []
        self._lock:         threading.Lock   = threading.Lock()
        self._condition:    threading.Condition = threading.Condition(self._lock)
        self._tasks:        dict[str, Task]  = {}
        self._running:      bool             = False
        self._worker_thread: threading.Thread | None = None
        self._max_concurrent = max_concurrent
        self._active_count   = 0
        self._executor       = None

    def _get_executor(self):
        if self._executor is None:
            from agent.executor import AgentExecutor
            self._executor = AgentExecutor()
        return self._executor

    def _write_task_row(self, task_id: str, **fields) -> None:
        """Write-through UPDATE to the tasks table — durability only, never
        read back by get_status()/get_all_statuses() (those stay on the
        in-memory dict, see below). `fields` keys are always internal
        column names we control, never user input."""
        if not fields:
            return
        fields["updated_at"] = time.time()
        set_clause = ", ".join(f"{col} = ?" for col in fields)
        values     = list(fields.values()) + [task_id]
        conn = get_conn()
        with conn:
            conn.execute(f"UPDATE tasks SET {set_clause} WHERE task_id = ?", values)

    def start(self) -> None:
        if self._running:
            return
        _reconcile_interrupted_tasks()
        _reconcile_orphaned_approvals()
        self._running      = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="AgentTaskQueue"
        )
        self._worker_thread.start()
        _log.info("Started")

    def stop(self) -> None:
        self._running = False
        with self._condition:
            self._condition.notify_all()
        _log.info("Stopped")

    def submit(
        self,
        goal:        str,
        priority:    TaskPriority = TaskPriority.NORMAL,
        speak:       Callable | None = None,
        on_complete: Callable | None = None,
        submitted_interactively: bool = True,
        caller_class: str = "desktop",
    ) -> str:

        task_id = str(uuid.uuid4())[:8]
        task    = Task(
            priority    = priority.value,
            created_at  = time.time(),
            task_id     = task_id,
            goal        = goal,
            speak       = speak,
            on_complete = on_complete,
            submitted_interactively = submitted_interactively,
            caller_class = caller_class,
        )

        with self._condition:
            self._queue.append(task)
            self._queue.sort(key=lambda t: (t.priority, t.created_at))
            self._tasks[task_id] = task
            self._condition.notify()

        conn = get_conn()
        with conn:
            conn.execute(
                "INSERT INTO tasks (task_id, goal, priority, status, result, error, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (task_id, goal, task.priority, task.status.value, None, "", task.created_at, task.created_at),
            )

        _log.info("Task queued: %s", goal[:60], extra={"task_id": task_id})
        return task_id

    def cancel(self, task_id: str) -> bool:

        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                return False

            task.cancel_flag.set()
            task.status = TaskStatus.CANCELLED
            _log.info("Task cancelled", extra={"task_id": task_id})

        self._write_task_row(task_id, status=TaskStatus.CANCELLED.value)
        return True

    def get_status(self, task_id: str) -> dict | None:
        """Reads the in-memory dict only — the DB is written-through for
        durability/history, not read-through here (unchanged from before
        the persistence layer landed)."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            return {
                "task_id": task.task_id,
                "goal":    task.goal,
                "status":  task.status.value,
                "result":  task.result,
                "error":   task.error,
            }

    def get_all_statuses(self) -> list[dict]:
        """Same in-memory-only read path as get_status() — see there."""
        with self._lock:
            return [
                {
                    "task_id": t.task_id,
                    "goal":    t.goal[:50],
                    "status":  t.status.value,
                }
                for t in self._tasks.values()
            ]

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._queue if t.status == TaskStatus.PENDING)

    def _worker_loop(self) -> None:
        while self._running:
            task = None

            with self._condition:
                while self._running and not self._next_task():
                    self._condition.wait(timeout=1.0)
                task = self._next_task()
                if task:
                    task.status = TaskStatus.RUNNING
                    self._active_count += 1
                    try:
                        self._queue.remove(task)
                    except ValueError:
                        pass

            if task:
                self._write_task_row(task.task_id, status=TaskStatus.RUNNING.value)
                threading.Thread(
                    target=self._run_task,
                    args=(task,),
                    daemon=True,
                    name=f"AgentTask-{task.task_id}"
                ).start()

    def _next_task(self) -> Task | None:
        if self._active_count >= self._max_concurrent:
            return None
        for task in self._queue:
            if task.status == TaskStatus.PENDING and not task.cancel_flag.is_set():
                return task
        return None

    def _run_task(self, task: Task) -> None:
        _log.info("Running: %s", task.goal[:60], extra={"task_id": task.task_id})
        try:
            executor = self._get_executor()
            result   = executor.execute(
                goal        = task.goal,
                speak       = task.speak,
                cancel_flag = task.cancel_flag,
                task_id     = task.task_id,
                submitted_interactively = task.submitted_interactively,
                caller_class = task.caller_class,
            )

            with self._lock:
                if task.cancel_flag.is_set():
                    task.status = TaskStatus.CANCELLED
                else:
                    failure_detail = _last_terminal_failure_detail(task.task_id)
                    if failure_detail is not None:
                        # See _last_terminal_failure_detail's docstring —
                        # execute() returned normally (no exception), but
                        # the real, durable outcome was a step the task
                        # ultimately gave up on (a denial, a hard-deny,
                        # etc.), not a genuine completion.
                        task.status = TaskStatus.FAILED
                        task.error  = failure_detail
                    else:
                        task.status = TaskStatus.COMPLETED
                        task.result = result
                self._active_count -= 1

            self._write_task_row(
                task.task_id,
                status = task.status.value,
                result = task.result if isinstance(task.result, str) else str(task.result) if task.result is not None else None,
                # Written on this path too now (previously only the
                # except-Exception path below persisted it) — a denial's
                # error detail must survive in the durable `tasks` row,
                # not just the in-memory Task object get_status() reads.
                # "" (not None) for the non-failed case, matching
                # submit()'s own INSERT convention for a task with no error.
                error  = task.error if task.status == TaskStatus.FAILED else "",
            )

            if task.on_complete and not task.cancel_flag.is_set():
                try:
                    task.on_complete(task.task_id, result)
                except Exception as e:
                    _log.warning("on_complete callback error: %s", e, extra={"task_id": task.task_id})

            _log.info("Completed", extra={"task_id": task.task_id})

        except Exception as e:
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error  = str(e)
                self._active_count -= 1
            self._write_task_row(task.task_id, status=TaskStatus.FAILED.value, error=task.error)
            _log.error("Failed: %s", e, extra={"task_id": task.task_id})

        with self._condition:
            self._condition.notify()

_queue        = TaskQueue()
_queue_started = False
_queue_lock    = threading.Lock()


def get_queue() -> TaskQueue:
    global _queue_started
    with _queue_lock:
        if not _queue_started:
            _queue.start()
            _queue_started = True
    return _queue
