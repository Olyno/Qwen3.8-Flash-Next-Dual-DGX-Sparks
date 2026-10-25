import copy
import importlib.util
import json
import os
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "score_proxy", os.path.join(ROOT, "proxy", "score_proxy.py"))
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)

DEPLOY = "---\nname: deploy\ndescription: Ship services safely.\n---\n\n# Deploy\n"
DEBUG = "# Debug\n\nTrace failures to their root cause.\n"
UNRELATED = "---\ndescription: Paint watercolor landscapes.\n---\n"

SKILL_VECTORS = {
    "deploy": [1.0, 0.0],
    "debug": [0.6, 0.8],
    "unrelated": [0.0, 1.0],
}
QUERY_TEXT = "how do I fix the deploy pipeline?"
QUERY_VECTOR = [1.0, 0.05]


def fake_embed(texts):
    out = []
    for text in texts:
        if text.startswith("how do I fix"):
            out.append(QUERY_VECTOR)
        else:
            out.append(SKILL_VECTORS[text.split(":", 1)[0]])
    return out


def boom(texts):
    raise OSError("embedding endpoint down")


def write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def user_req(text=QUERY_TEXT):
    return {"model": "m", "messages": [{"role": "user", "content": text}]}


class SkillParseTest(unittest.TestCase):
    def test_frontmatter_description(self):
        self.assertEqual(sp.skill_description(
            "---\nname: deploy\ndescription: Ship services safely.\n---\n"
            "\n# Deploy\n\nBody.\n"),
            "Ship services safely.")

    def test_frontmatter_quotes_stripped(self):
        self.assertEqual(sp.skill_description(
            '---\ndescription: "Ship: safely"\n---\n'),
            "Ship: safely")

    def test_fallback_first_non_heading_paragraph(self):
        self.assertEqual(sp.skill_description(
            "# Title\n\nIntro line one\nline two.\n\nSecond paragraph."),
            "Intro line one line two.")

    def test_fallback_skips_frontmatter_without_description(self):
        self.assertEqual(sp.skill_description(
            "---\nname: deploy\n---\n\n# Deploy\n\nBody text."),
            "Body text.")

    def test_fallback_truncated(self):
        desc = sp.skill_description("# Title\n\n" + "x" * 300)
        self.assertEqual(len(desc), sp.SKILL_DESC_MAX)
        self.assertEqual(desc, "x" * sp.SKILL_DESC_MAX)


class SkillIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.deploy_path = write(self.root, os.path.join("deploy", "SKILL.md"),
                                 DEPLOY)
        self.debug_path = write(self.root, "debug.md", DEBUG)
        write(self.root, os.path.join("unrelated", "SKILL.md"), UNRELATED)

    def tearDown(self):
        self.tmp.cleanup()

    def test_layouts_names_and_paths(self):
        ranked = sp.SkillIndex(self.root, fake_embed).rank(QUERY_TEXT)
        got = {name: path for _, name, _, path in ranked}
        self.assertEqual(got, {"deploy": self.deploy_path,
                               "debug": self.debug_path})

    def test_flat_layout_only(self):
        with tempfile.TemporaryDirectory() as root:
            path = write(root, "deploy.md",
                         "---\ndescription: Flat deploy.\n---\n")
            ranked = sp.SkillIndex(root, fake_embed).rank(QUERY_TEXT)
            self.assertEqual([r[1] for r in ranked], ["deploy"])
            self.assertEqual(ranked[0][3], path)

    def test_ranking_and_threshold_order(self):
        ranked = sp.SkillIndex(self.root, fake_embed).rank(QUERY_TEXT)
        self.assertEqual([r[1] for r in ranked], ["deploy", "debug"])
        self.assertGreater(ranked[0][0], ranked[1][0])
        only = sp.SkillIndex(self.root, fake_embed, min_sim=0.7).rank(QUERY_TEXT)
        self.assertEqual([r[1] for r in only], ["deploy"])
        none = sp.SkillIndex(self.root, fake_embed, min_sim=0.999).rank(QUERY_TEXT)
        self.assertEqual(none, [])

    def test_top_k_caps_results(self):
        ranked = sp.SkillIndex(self.root, fake_embed, top_k=1).rank(QUERY_TEXT)
        self.assertEqual([r[1] for r in ranked], ["deploy"])

    def test_injects_block_as_trailing_user_message(self):
        req = user_req()
        changed, sim, count = sp.SkillIndex(self.root, fake_embed).apply(req)
        self.assertTrue(changed)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(sim, 0.998751, places=5)
        self.assertEqual(req["messages"][0]["role"], "user")
        self.assertEqual(req["messages"][-1]["role"], "user")
        self.assertEqual(req["messages"][-1]["content"], (
            sp.TAIL_FRAME + sp.SKILLS_MARKER + "\n"
            f"- deploy: {sp.skill_description(DEPLOY)} [{self.deploy_path}]\n"
            f"- debug: {sp.skill_description(DEBUG)} [{self.debug_path}]"))

    def test_existing_system_message_stays_at_the_head(self):
        req = {"model": "m", "messages": [
            {"role": "system", "content": "base"},
            {"role": "user", "content": QUERY_TEXT}]}
        changed, _, _ = sp.SkillIndex(self.root, fake_embed).apply(req)
        self.assertTrue(changed)
        self.assertEqual(req["messages"][0], {"role": "system",
                                              "content": "base"})
        self.assertEqual(req["messages"][-1]["role"], "user")
        self.assertTrue(req["messages"][-1]["content"]
                        .startswith(sp.TAIL_FRAME))
        self.assertEqual(req["messages"][-1]["content"]
                         .count(sp.SKILLS_MARKER), 1)

    def test_idempotent(self):
        index = sp.SkillIndex(self.root, fake_embed)
        req = user_req()
        self.assertTrue(index.apply(req)[0])
        first = req["messages"][-1]["content"]
        changed, _, _ = index.apply(req)
        self.assertFalse(changed)
        self.assertEqual(req["messages"][-1]["content"], first)
        self.assertEqual(first.count(sp.SKILLS_MARKER), 1)

    def test_marker_in_any_message_blocks_duplicate(self):
        req = {"model": "m", "messages": [
            {"role": "system", "content": "base"},
            {"role": "system", "content": sp.SKILLS_MARKER + "\n- old"},
            {"role": "user", "content": QUERY_TEXT}]}
        changed, _, _ = sp.SkillIndex(self.root, fake_embed).apply(req)
        self.assertFalse(changed)
        self.assertEqual(req["messages"][0]["content"], "base")

    def test_empty_user_message_noop(self):
        req = {"model": "m", "messages": [
            {"role": "system", "content": "base"}]}
        changed, sim, count = sp.SkillIndex(self.root, boom).apply(req)
        self.assertFalse(changed)
        self.assertEqual((sim, count), (0.0, 0))
        self.assertEqual(req["messages"][0]["content"], "base")

    def test_missing_directory_fails_open(self):
        req = user_req()
        index = sp.SkillIndex(os.path.join(self.root, "nope"), boom)
        changed, sim, count = index.apply(req)
        self.assertFalse(changed)
        self.assertEqual((sim, count), (0.0, 0))
        self.assertEqual(req["messages"][0]["role"], "user")

    def test_embed_failure_fails_open_and_caches_empty(self):
        index = sp.SkillIndex(self.root, boom)
        req = user_req()
        self.assertFalse(index.apply(req)[0])
        self.assertEqual(index._skills, [])
        # empty cache means no second scan/embed attempt
        self.assertFalse(index.apply(user_req())[0])

    def test_skill_scan_runs_once(self):
        calls = []

        def counting_embed(texts):
            calls.append(list(texts))
            return fake_embed(texts)

        index = sp.SkillIndex(self.root, counting_embed)
        index.apply(user_req())
        index.apply(user_req())
        self.assertEqual(len(calls), 3)        # scan batch + one query each
        self.assertEqual(len(calls[0]), 3)
        self.assertEqual([len(c) for c in calls[1:]], [1, 1])

    def test_pinned_skill_bypasses_threshold_and_inlines_body(self):
        with tempfile.TemporaryDirectory() as root:
            write(root, os.path.join("ponytail", "SKILL.md"),
                  "---\nname: ponytail\ndescription: Laziest fix.\n---\n\n"
                  "# Ponytail\n\nAsk if it needs to exist.\n")

            def embed(texts):
                return [[0.0, 1.0] if t.startswith("ponytail:")
                        else [1.0, 0.0] for t in texts]

            index = sp.SkillIndex(root, embed, min_sim=0.999,
                                  pin=("ponytail",), pin_chars=200)
            req = user_req()
            changed, _, count = index.apply(req)
            self.assertTrue(changed)
            self.assertEqual(count, 1)
            self.assertEqual(req["messages"][0]["role"], "system")
            content = req["messages"][0]["content"]
            self.assertIn("## ponytail", content)
            self.assertIn("Ask if it needs to exist.", content)
            self.assertNotIn("SKILL.md", content)

    def test_pinned_skill_skips_scan_and_listing(self):
        with tempfile.TemporaryDirectory() as root:
            write(root, os.path.join("ponytail", "SKILL.md"),
                  "---\nname: ponytail\ndescription: Laziest fix.\n---\n\n"
                  "# Ponytail\n\nAsk if it needs to exist.\n")
            write(root, "debug.md", DEBUG)
            calls = []

            def embed(texts):
                calls.append(list(texts))
                return [[1.0, 0.0] for _ in texts]

            index = sp.SkillIndex(root, embed, pin=("ponytail",))
            req = user_req()
            with mock.patch("os.listdir",
                            side_effect=AssertionError("directory scanned")):
                changed, _, count = index.apply(req)
            self.assertTrue(changed)
            self.assertEqual(count, 1)
            self.assertEqual(calls, [])       # no scan embedding, no query
            self.assertEqual(req["messages"][0]["role"], "system")
            content = req["messages"][0]["content"]
            self.assertIn("## ponytail", content)
            self.assertNotIn("debug", content)

    def test_changed_skill_set_leaves_stable_head_byte_identical(self):
        conversation = [
            {"role": "system", "content": "base policy"},
            {"role": "assistant", "content": "ready"},
            {"role": "user", "content": QUERY_TEXT},
        ]
        with tempfile.TemporaryDirectory() as base, \
                tempfile.TemporaryDirectory() as extra:
            for root in (base, extra):
                write(root, os.path.join("deploy", "SKILL.md"), DEPLOY)
                write(root, "debug.md", DEBUG)
            write(extra, os.path.join("unrelated", "SKILL.md"), UNRELATED)

            first = {"model": "m", "messages": copy.deepcopy(conversation)}
            sp.SkillIndex(base, fake_embed).apply(first)

            real_listdir = os.listdir

            def reversed_listdir(path):
                return list(reversed(real_listdir(path)))

            second = {"model": "m", "messages": copy.deepcopy(conversation)}
            with mock.patch("os.listdir", side_effect=reversed_listdir):
                sp.SkillIndex(extra, fake_embed).apply(second)

        head = len(conversation)
        self.assertEqual(
            json.dumps(first["messages"][:head], sort_keys=True).encode(),
            json.dumps(second["messages"][:head], sort_keys=True).encode())
        for req in (first, second):
            self.assertEqual(req["messages"][:head], conversation)
            self.assertEqual(len(req["messages"]), head + 1)
            self.assertEqual(req["messages"][-1]["role"], "user")
            self.assertIn(sp.SKILLS_MARKER, req["messages"][-1]["content"])
            self.assertNotIn(sp.SKILLS_MARKER,
                             json.dumps(req["messages"][:head]))


class SkillWiringTest(unittest.TestCase):
    def test_env_unset_feature_off(self):
        srv = sp.make_server(0, "http://127.0.0.1:1")
        try:
            self.assertIsNone(srv.skill_index)
        finally:
            srv.server_close()

    def test_pin_works_without_an_embedding_service(self):
        with tempfile.TemporaryDirectory() as root:
            write(root, os.path.join("ponytail", "SKILL.md"),
                  "---\nname: ponytail\ndescription: Laziest fix.\n---\n\n"
                  "# Ponytail\n\nAsk if it needs to exist.\n")
            srv = sp.make_server(0, "http://127.0.0.1:1",
                                 skills_dir=root, skills_pin=("ponytail",))
            try:
                self.assertIsInstance(srv.skill_index, sp.SkillIndex)
                self.assertIsNone(srv.skill_index.embed_fn)
                req = {"model": "m", "messages": [
                    {"role": "user", "content": "fix the failing test"}]}
                self.assertTrue(srv.skill_index.apply(req)[0])
                self.assertIn("## ponytail", json.dumps(req["messages"]))
            finally:
                srv.server_close()

    def test_listing_still_needs_the_router(self):
        with tempfile.TemporaryDirectory() as root:
            srv = sp.make_server(0, "http://127.0.0.1:1", skills_dir=root)
            try:
                self.assertIsNone(srv.skill_index)
            finally:
                srv.server_close()


        with tempfile.TemporaryDirectory() as root:
            srv = sp.make_server(0, "http://127.0.0.1:1",
                                 router_url="http://127.0.0.1:1",
                                 skills_dir=root, skills_top_k=2,
                                 skills_min_sim=0.4,
                                 skills_pin=("ponytail",))
            try:
                self.assertIsInstance(srv.skill_index, sp.SkillIndex)
                self.assertEqual(srv.skill_index.top_k, 2)
                self.assertEqual(srv.skill_index.min_sim, 0.4)
                self.assertEqual(srv.skill_index.pin, ("ponytail",))
            finally:
                srv.server_close()


if __name__ == "__main__":
    unittest.main()
