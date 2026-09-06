"""Fast checks for the failure paths, with no network and no quota.

Run from the project root:

    python tests/check.py

These exist because the only way to find out whether this pipeline worked used
to be to run it: twenty-five minutes, real generation quota, and a verdict that
arrived at the end. Bugs surfaced ten steps after the mistake that caused them.
Everything here runs in about a second against fake HTTP, and covers the paths
that only happen when someone else's server misbehaves -- which is exactly the
part a real run exercises least reliably.

Nothing here talks to a provider, writes into output/, or needs a key.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from omniclip.core import jobstore  # noqa: E402
from omniclip.core.jobstore import Job, SceneRecord  # noqa: E402
from omniclip.core import video_rotator as vr  # noqa: E402
from omniclip.frontend import queue_store as q, worker  # noqa: E402

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# --------------------------------------------------------------- fake HTTP

QUOTA_BODY = '{"error":"free user quota exceeded, please upgrade your plan"}'


class FakeResponse:
    def __init__(self, status_code, text='{}', payload=None):
        self.status_code = status_code
        self.text = text
        self.content = b"\x00" * 512
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class FakeStream:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_bytes(self, size):
        yield b"\x00" * 512


class FakeAgnes:
    """Stands in for the service. `spent` names the keys that are out of plan."""

    def __init__(self, spent=()):
        self.spent = set(spent)
        self.submits = []

    def install(self):
        vr.httpx.post = self.post
        vr.httpx.get = self.get
        vr.httpx.stream = lambda *a, **k: FakeStream()
        return self

    def post(self, url, headers=None, json=None, timeout=None):
        key = headers["Authorization"].split()[-1]
        self.submits.append(key)
        if key in self.spent:
            return FakeResponse(429, QUOTA_BODY)
        if "images" in url:
            return FakeResponse(200, '{}', {"data": [{"url": "http://x/s.jpg"}]})
        return FakeResponse(200, '{"video_id":"v1"}', {"video_id": "v1"})

    def get(self, url, params=None, headers=None, timeout=None,
            follow_redirects=None):
        return FakeResponse(200, '{}',
                            {"status": "completed", "url": "http://x/c.mp4"})


def provider(keys, spent=()):
    """An Agnes provider wired to the fake service, paced fast enough to test."""
    FakeAgnes(spent).install()
    p = vr.AgnesProvider(api_key=list(keys), rpm=600, pool_rpm=600)
    p.poll_interval = 0.0
    return p


# ------------------------------------------------------------- the key ring


@check
def key_ring_retires_keys_one_at_a_time():
    ring = vr.KeyRing(["a", "b", "c"], cooldown=0.0)
    assert len(ring) == 3 and ring.total == 3
    assert ring.retire("a", "429") == 2
    assert ring.peek_ready() == "b", "a retired key must not be handed out"
    assert len(ring) == 2, "live count drops"
    assert ring.total == 3, "the roster does not shrink"
    ring.retire("b", "429")
    ring.retire("c", "429")
    try:
        ring.acquire()
    except vr.QuotaExhausted:
        pass
    else:
        raise AssertionError("an empty ring must raise QuotaExhausted")


@check
def one_spent_key_does_not_end_the_run():
    """The bug this whole exercise started from: a single 429 killed all 25."""
    p = provider(["a", "b", "c"], spent=("a", "b"))
    with tempfile.TemporaryDirectory() as tmp:
        asset = p.fetch("a quiet street", 4.0, Path(tmp))
        assert asset is not None, "key c should have produced the clip"
    assert len(p.keys) == 1, "only the two spent keys are retired"
    assert not p._exhausted.is_set(), "a live key remains, the run is not over"


@check
def the_run_ends_only_when_every_key_is_spent():
    p = provider(["a", "b"], spent=("a", "b"))
    with tempfile.TemporaryDirectory() as tmp:
        try:
            p.fetch("a quiet street", 4.0, Path(tmp))
        except vr.QuotaExhausted as exc:
            assert "out of capacity" in str(exc)
        else:
            raise AssertionError("expected QuotaExhausted once all keys went")
    assert p._exhausted.is_set()


@check
def stills_also_walk_past_a_spent_key():
    p = provider(["a", "b", "c"], spent=("a",))
    with tempfile.TemporaryDirectory() as tmp:
        got = p.make_still("a boy on a step", Path(tmp))
        assert got is not None and Path(got[1]).exists()
    assert len(p.keys) == 2


# --------------------------------------------------- the type, not the text


@check
def quota_stays_quota_through_the_provider_chain():
    """A caller must not have to read a message to learn it is out of quota."""

    class Spent:
        name = "agnes"

        def fetch(self, *a, **k):
            raise vr.QuotaExhausted("the plan is spent")

    with tempfile.TemporaryDirectory() as tmp:
        sourcer = vr.VisualSourcer([Spent()], tmp)
        try:
            sourcer.fetch_one("a very long scene prompt " * 20, 4.0)
        except vr.QuotaExhausted as exc:
            # And the message stays short: the thousand-character prompt in it
            # was what made the log line that got half-written and misread.
            assert len(str(exc)) < 300, f"message too long: {len(str(exc))}"
        else:
            raise AssertionError("QuotaExhausted must survive fetch_one")


@check
def an_ordinary_failure_is_still_an_ordinary_failure():
    class Broken:
        name = "pexels"

        def fetch(self, *a, **k):
            raise RuntimeError("connection reset")

    with tempfile.TemporaryDirectory() as tmp:
        sourcer = vr.VisualSourcer([Broken()], tmp)
        try:
            sourcer.fetch_one("a street", 4.0)
        except vr.QuotaExhausted:
            raise AssertionError("a network error is not a spent quota")
        except vr.VisualError:
            pass


# ------------------------------------------------------- the stop record


def job_with(tmp, reason=None, scenes=2, with_clips=0, log=None, **stop_extra):
    """A saved job in `tmp`, optionally with a stop record and a run log."""
    out = Path(tmp)
    job = Job(directory=out, url="http://example/v")
    job.scenes = [SceneRecord(index=i) for i in range(scenes)]
    for i in range(with_clips):
        clip = out / f"clip{i}.mp4"
        clip.write_bytes(b"x")
        job.scenes[i].asset_path = str(clip)
    job.save()
    if reason:
        job.mark_stop(reason, "detail for a person", **stop_extra)
    if log is not None:
        (out / "run.log").write_text(log, encoding="utf-8")
    return job


@check
def stop_record_round_trips():
    with tempfile.TemporaryDirectory() as tmp:
        job = job_with(tmp, reason="quota")
        again = Job.load(tmp)
        assert again.stop["reason"] == "quota"
        again.clear_stop()
        assert Job.load(tmp).stop == {}, "a new attempt starts with no ending"


@check
def an_unknown_reason_is_recorded_as_an_error():
    with tempfile.TemporaryDirectory() as tmp:
        job = job_with(tmp)
        job.mark_stop("banana", "nonsense")
        assert job.stop["reason"] == "error"
        assert job.stop["given_reason"] == "banana", "the original is kept"


@check
def a_resumed_job_forgets_the_previous_ending():
    with tempfile.TemporaryDirectory() as tmp:
        job_with(tmp, reason="quota")
        resumed = jobstore.open_job(tmp, "http://example/v", {})
        assert resumed.stop == {}, "last night's quota stop is not today's"


# ------------------------------------------------ what the queue decides


@check
def quota_is_a_hold_not_a_failure():
    with tempfile.TemporaryDirectory() as tmp:
        job_with(tmp, reason="quota", scenes=12, with_clips=5)
        state, message, retry_at = worker.outcome_for(Path(tmp))
        assert state == q.HOLDING, state
        assert "5/12" in message, message
        assert retry_at > 0, "a hold must say when to come back"


@check
def an_error_is_a_failure():
    with tempfile.TemporaryDirectory() as tmp:
        job_with(tmp, reason="error")
        state, message, _ = worker.outcome_for(Path(tmp))
        assert state == q.FAILED, state
        assert "detail for a person" in message


@check
def a_pause_waits_for_a_person():
    with tempfile.TemporaryDirectory() as tmp:
        job_with(tmp, reason="paused", stage="stills")
        state, message, _ = worker.outcome_for(Path(tmp))
        assert state == q.REVIEW, state
        assert "stills" in message


@check
def a_half_written_log_line_no_longer_decides_the_outcome():
    """The exact bug that filed a capacity stop as a failure.

    The marker the old code matched sat 1,255 characters into a 1,390-character
    line. When the worker read the log before that line finished flushing, the
    match failed and a job that was only out of capacity was reported as dead.
    Here the log is truncated in precisely that way, and the answer still comes
    out right, because it now comes from a field.
    """
    truncated = "FAILED: VisualError: No clip for 'Medium shot from the side."
    assert "QuotaExhausted" not in truncated
    with tempfile.TemporaryDirectory() as tmp:
        job_with(tmp, reason="quota", scenes=12, with_clips=5, log=truncated)
        state, _, _ = worker.outcome_for(Path(tmp))
        assert state == q.HOLDING, f"expected a hold, got {state}"


@check
def an_old_job_without_a_record_still_reads_its_log():
    """Jobs written before the stop record exists must not all become failures."""
    log = "FAILED: VisualError: No clip (agnes: QuotaExhausted: out of capacity)"
    with tempfile.TemporaryDirectory() as tmp:
        job_with(tmp, reason=None, scenes=4, with_clips=1, log=log)
        state, _, _ = worker.outcome_for(Path(tmp))
        assert state == q.HOLDING, f"legacy fallback broke: {state}"


# ------------------------------------------------------ the two front doors


@check
def the_cli_and_the_page_share_one_stock_default():
    """They used to carry opposite defaults, so the same job behaved
    differently depending on which one started it."""
    from omniclip.utils.config import load_settings

    cfg = load_settings()
    assert "allow_stock" in cfg["visuals"], "the shared default must exist"

    source = (ROOT / "omniclip" / "frontend" / "app.py").read_text(encoding="utf-8")
    assert 'cfg["visuals"].get("allow_stock"' in source, \
        "the page must read the shared default, not hardcode one"
    source = (ROOT / "omniclip" / "cli.py").read_text(encoding="utf-8")
    assert 'cfg["visuals"].get("allow_stock"' in source, \
        "the CLI must read the shared default, not hardcode one"


@check
def the_page_builds_the_command_it_claims_to():
    from omniclip.frontend import runner

    command = runner.build_command(
        "http://example/v", Path("out"), {"no_stock": True, "aspect": "9:16"})
    assert "--no-stock" in command and "--aspect" in command
    assert "9:16" in command


# ------------------------------------------------------------------ runner


def main() -> int:
    failed = []
    for fn in CHECKS:
        name = fn.__name__.replace("_", " ")
        try:
            fn()
        except Exception:
            failed.append(fn.__name__)
            print(f"  FAIL  {name}")
            print("        " + traceback.format_exc().strip().replace("\n", "\n        "))
        else:
            print(f"  ok    {name}")

    print()
    if failed:
        print(f"{len(failed)} of {len(CHECKS)} checks failed: {', '.join(failed)}")
        return 1
    print(f"all {len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
