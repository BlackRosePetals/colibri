"""machine_info() must report the machine's real RAM on win32 (#1042).

ram_gb sizes the eviction write in evict_cache(): a hardcoded 8.0 on a
128 GB box writes 9 GB, evicts nothing, and the run labelled "cold" is
measured warm and published as cold. These tests run on any host by
mocking sys.platform and the two win32 probes.
"""
import ctypes
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from family_registry import FAMILIES
from tools import datapoint

GB = 1073741824


class MachineInfoWin32Test(unittest.TestCase):
    def _win32_info(self, memstatus_ok, total_phys=128 * GB):
        def fake_memstatus(argp):
            if not memstatus_ok:
                return 0
            argp._obj.ullTotalPhys = total_phys
            return 1

        windll = mock.MagicMock()
        windll.kernel32.GlobalMemoryStatusEx.side_effect = fake_memstatus
        with mock.patch.object(sys, "platform", "win32"), \
             mock.patch.object(ctypes, "windll", windll, create=True):
            return datapoint.machine_info()

    def test_win32_reports_real_ram(self):
        info = self._win32_info(memstatus_ok=True)
        self.assertAlmostEqual(info["ram_gb"], 128.0)
        self.assertEqual(info["ram"], "128 GB")
        self.assertTrue(info["os"].startswith("Windows"))

    def test_win32_probe_failure_falls_back(self):
        """A failed probe keeps the old conservative default rather than 0.0 —
        an eviction write sized from 0 GB would silently evict nothing."""
        info = self._win32_info(memstatus_ok=False)
        self.assertEqual(info["ram_gb"], 8.0)
        self.assertEqual(info["ram"], "?")

    def test_unknown_platform_keeps_fallback(self):
        with mock.patch.object(sys, "platform", "freebsd14"):
            info = datapoint.machine_info()
        self.assertEqual(info["ram_gb"], 8.0)
        self.assertEqual(info["ram"], "?")


class PersistentDatapointTest(unittest.TestCase):
    def setUp(self):
        self.family = SimpleNamespace(
            id="deepseek_v4",
            display_name="DeepSeek V4",
            has_gateway_adapter=True,
            build_target="deepseek_v4",
        )
        self.instances = []

    def _runtime(self, engine_type=None):
        instances = self.instances

        class FakeEngine:
            def __init__(self, executable, model, cap, max_tokens, env, kv_slots,
                         family):
                self.executable = executable
                self.model = model
                self.cap = cap
                self.max_tokens = max_tokens
                self.env = env
                self.kv_slots = kv_slots
                self.family = family
                self.calls = []
                self.closed = False
                self.tiers = {"vram": 2, "ram": 3, "disk": 4,
                              "vram_gb": 1.5, "ram_gb": 2.5}
                self.hwinfo = {"cores": 16}
                self.profile = []
                self.profile_seq = 0
                instances.append(self)

            def generate(self, prompt, max_tokens, temperature, top_p, on_text,
                         cache_slot=0):
                self.calls.append((prompt, max_tokens, temperature, top_p, cache_slot))
                on_text("token")
                self.profile.append({"expert_disk_s": 0.1,
                                     "expert_wait_s": 0.2,
                                     "expert_matmul_s": 0.3,
                                     "attention_s": 0.4,
                                     "lm_head_s": 0.5})
                self.profile_seq += 1
                return {"completion_tokens": max_tokens,
                        "tokens_per_second": float(len(self.calls)),
                        "cache_hit_percent": float(len(self.calls) * 10),
                        "rss_gb": 12.5,
                        "prompt_tokens": 17,
                        "length_limited": True}

            def close(self):
                self.closed = True

        return SimpleNamespace(
            ARCH=None,
            Engine=engine_type or FakeEngine,
            resolve_model=lambda _snap: SimpleNamespace(descriptor=self.family),
            default_engine=lambda _family: "/unused/default-engine",
            render_chat_for_arch=lambda messages, enable_thinking: (
                f"rendered:{messages[0]['content']}:{enable_thinking}"),
        )

    def test_default_campaign_reuses_one_engine_for_every_request(self):
        runtime = self._runtime()
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "deepseek_v4"
            executable.touch()
            with mock.patch.dict(os.environ, {}, clear=True):
                campaign = datapoint.run_persistent_engine(
                    str(executable), "/model", "hello", 8, 1, 3, 16,
                    memory_gb=64, runtime=runtime, physical_cores=8)

        self.assertEqual(len(self.instances), 1)
        engine = self.instances[0]
        self.assertEqual(len(engine.calls), 5)  # cold + warm-up + 3 measured
        self.assertTrue(all(call[0] == "rendered:hello:False" for call in engine.calls))
        self.assertTrue(all(call[4] == 0 for call in engine.calls))
        self.assertTrue(engine.closed)
        self.assertEqual(engine.env["RAM_GB"], "64")
        self.assertEqual(engine.env["COLI_TEMP"], "0")
        self.assertNotIn("OMP_NUM_THREADS", engine.env)  # V4 owns its thread split
        self.assertEqual(runtime.ARCH, "deepseek_v4")
        self.assertEqual(len(campaign["cold"]), 1)
        self.assertEqual(len(campaign["warmup"]), 1)
        self.assertEqual(len(campaign["warm"]), 3)
        self.assertEqual([row["tok_s"] for row in campaign["warm"]], [3.0, 4.0, 5.0])
        self.assertEqual(campaign["warm"][0]["tokens"], 8)
        self.assertEqual(campaign["warm"][0]["prompt_tokens"], 17)

    def test_engine_is_closed_when_a_request_fails(self):
        instances = self.instances

        class FailingEngine:
            def __init__(self, *_args, **_kwargs):
                self.profile_seq = 0
                self.profile = []
                self.closed = False
                instances.append(self)

            def generate(self, *_args, **_kwargs):
                raise RuntimeError("generation failed")

            def close(self):
                self.closed = True

        runtime = self._runtime(FailingEngine)
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "deepseek_v4"
            executable.touch()
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                datapoint.run_persistent_engine(
                    str(executable), "/model", "hello", 8, 1, 3, 16,
                    runtime=runtime, physical_cores=8)
        self.assertTrue(self.instances[0].closed)

    def test_trailing_profile_after_done_is_attached_to_the_request(self):
        class DelayedProfileEngine:
            profile_seq = 0
            profile = []

            def generate(self, _prompt, max_tokens, _temperature, _top_p,
                         on_text, cache_slot=0):
                self.asserted_slot = cache_slot
                on_text("token")

                def emit_profile():
                    self.profile = [{"expert_disk_s": 1.25}]
                    self.profile_seq += 1

                self.timer = threading.Timer(0.01, emit_profile)
                self.timer.start()
                return {"completion_tokens": max_tokens,
                        "tokens_per_second": 2.0,
                        "cache_hit_percent": 50.0,
                        "rss_gb": 4.0,
                        "prompt_tokens": 3,
                        "length_limited": True}

        engine = DelayedProfileEngine()
        result = datapoint._measure_persistent_request(engine, "prompt", 8)
        engine.timer.join()
        self.assertEqual(engine.asserted_slot, 0)
        self.assertEqual(result["profile"]["expert_disk_s"], 1.25)

    def test_sister_engines_use_physical_cores_without_overriding_user_value(self):
        family = SimpleNamespace(id="qwen36")
        with mock.patch.dict(os.environ, {}, clear=True):
            env = datapoint._persistent_environment(family, 128, 32, 16)
            self.assertEqual(env["OMP_NUM_THREADS"], "16")
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "6"}, clear=True):
            env = datapoint._persistent_environment(family, 128, 32, 16)
            self.assertEqual(env["OMP_NUM_THREADS"], "6")

    def test_persistent_mode_rejects_the_cli_wrapper(self):
        with self.assertRaisesRegex(ValueError, "engine binary"):
            datapoint._persistent_executable(None, self.family, "./coli")

    def test_parser_defaults_to_persistent_campaign(self):
        args = datapoint.build_parser().parse_args(["--snap", "/model"])
        self.assertEqual(args.mode, "persistent")
        self.assertEqual(args.warmup_runs, 1)
        self.assertEqual(args.warm_runs, 3)
        self.assertIsNone(args.engine)

    def test_every_registered_family_has_the_shared_persistent_adapter(self):
        expected = {"glm", "inkling", "kimi", "olmoe", "qwen36", "deepseek_v4"}
        self.assertTrue(expected.issubset({family.id for family in FAMILIES}))
        self.assertTrue(all(family.has_gateway_adapter for family in FAMILIES))


if __name__ == "__main__":
    unittest.main()
