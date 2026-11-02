"""The draft layer switch: "mtp": false in a run config starts the engine without --mtp (serve/server.py engine_args), however the
arguments got there, and strata_control/draft_layer.py is the one rule setup, the launcher and the server share.

    python -m unittest serve.test_draft_layer
"""
import contextlib
import io
import unittest

from serve import runconfig
from serve.server import engine_args
from strata_control import draft_layer

ARGS = ["--pack", "/p", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "/m/rt", "--max-context", "32768", "--batch-mtp", "--kv", "int8"]


class DraftLayer(unittest.TestCase):
    def test_strip_keeps_everything_but_the_draft_layer(self):
        out, removed = draft_layer.strip(ARGS)
        self.assertEqual(out, ["--pack", "/p", "--spec", "4", "--max-context", "32768", "--kv", "int8"])
        self.assertEqual(removed, ["--spec-min-p", "--mtp", "--batch-mtp"])
        self.assertEqual(draft_layer.strip(["--mtp", "2", "--spec", "2"])[0], ["--spec", "2"])      # Qwen3.6: a draft count

    def test_only_an_explicit_false_turns_it_off(self):
        for cfg in ({}, {"mtp": True}, {"mtp": None}, {"mtp": 0}, {"mtp": "off"}, None):
            self.assertFalse(draft_layer.is_off(cfg), cfg)
        self.assertTrue(draft_layer.is_off({"mtp": False}))

    def test_the_server_applies_it_at_every_start(self):
        on = engine_args({"args": list(ARGS)})
        self.assertIn("--mtp", on)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            off = engine_args({"args": list(ARGS), "mtp": False})
        for flag in ("--mtp", "--spec-min-p", "--batch-mtp"):
            self.assertNotIn(flag, off)
        self.assertEqual(off[off.index("--spec") + 1], "4")
        self.assertIn("without the draft layer", out.getvalue())

    def test_the_web_settings_page_lists_it(self):
        cfg = {"args": []}
        self.assertTrue(any(k["key"] == "mtp" for k in runconfig.view(cfg, "strata-x.json")["keys"]))
        new, changed = runconfig.apply(cfg, {"mtp": False})
        self.assertIs(new["mtp"], False)
        self.assertEqual(changed, ["mtp"])
        with self.assertRaises(ValueError):
            runconfig.apply(cfg, {"mtp": "off"})

    def test_three_level_cannot_silently_drop_mtp_or_enable_batching(self):
        args = ["--pack", "/p", "--mtp", "/m/rt", "--three-level"]
        self.assertEqual(engine_args({"args": args, "drafting": "auto"}), args)
        self.assertEqual(engine_args({"args": args, "gpu": [0], "layer_split": "auto"}), args)
        for config in ({"mtp": False}, {"drafting": "off"}, {"drafting": "dflash2"},
                       {"parallel": 2}, {"gpu": [0, 1]}, {"backend": "hip"},
                       {"replicas": [[0]]}, {"split_serve": {"prefill_gpu": 0}},
                       {"vision": {}}, {"vision": {"gpu": True}}, {"sampling": {"temperature": 0.7}},
                       {"reasoning_close_retry": True}, {"reasoning_budget_tokens": 20},
                       {"reasoning_loop_recovery": "recover"}):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "three-level"):
                engine_args({"args": args, **config})
        for extra in (["--mtp", "0"], ["--three-level-loops", "7"], ["--three-level-depth", "0"],
                      ["--three-level-depth", "2.5"], ["--preverify-top-h", "11"], ["--batch", "1"],
                      ["--three-level-loops"]):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "three-level"):
                engine_args({"args": args + extra})
        with self.assertRaisesRegex(ValueError, "three-level"):
            engine_args({"args": args + ["--preverify-top-h", "9"], "arch": "qwen35moe"})
        with self.assertRaisesRegex(ValueError, "three-level"):
            engine_args({"args": args, "user_args": ["--mtp", "0"]})
        self.assertIn("--three-level", engine_args({"args": args, "sampling": {"temperature": 0}}))


if __name__ == "__main__":
    unittest.main()
