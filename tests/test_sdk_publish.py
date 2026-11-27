"""The mod SDK is published in one file, sdk-publish.yml, which every publisher calls (#4022, #4108)."""
import unittest

from tests.test_ci_concurrency import ROOT
from tools.ci import vendored_yaml

yaml = vendored_yaml.load()

WORKFLOWS = ROOT / ".github/workflows"


def load(name):
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


class SdkPublishTests(unittest.TestCase):
    def test_only_sdk_publish_releases_an_sdk_and_every_caller_hands_it_the_key(self):
        publishers = sorted(path.name for path in WORKFLOWS.glob("*.yml") if "Release sdk-<sha>" in path.read_text())
        self.assertEqual(publishers, ["sdk-publish.yml"])
        callers = {}
        for path in WORKFLOWS.glob("*.yml"):
            for name, job in load(path.name)["jobs"].items():
                if job.get("uses") == "./.github/workflows/sdk-publish.yml":
                    callers[path.name] = job
        self.assertEqual(sorted(callers), ["daily.yml", "nightly-build.yml", "prebuilt-game.yml"])
        for name, job in callers.items():
            with self.subTest(caller=name):
                self.assertEqual(job["secrets"], "inherit")
                self.assertEqual(job["permissions"]["contents"], "write")
                self.assertEqual(set(job["with"]) - {"draft"}, {"commit", "publish"})
        # Only a Dev build drafts its SDK; the daily run and the hand dispatch publish at once.
        self.assertEqual({name for name, job in callers.items() if "draft" in job["with"]}, {"nightly-build.yml"})

    def test_a_dev_build_publishes_its_sdk_only_when_it_is_no_dry_run(self):
        job = load("nightly-build.yml")["jobs"]["sdk-publish"]
        self.assertEqual(job["with"]["publish"], "${{ needs.pick.outputs.publish == 'true' }}")
        self.assertEqual(job["with"]["draft"], "true")
        self.assertIn("needs.sdk.outputs.needed == 'true'", job["if"])
        self.assertIn("inputs.kind == 'dev'", job["if"])

    def test_a_draft_sdk_gets_its_book_only_once_it_is_public(self):
        book = load("sdk-publish.yml")["jobs"]["book"]
        self.assertEqual(book["uses"], "./.github/workflows/modding-book-generated.yml")
        self.assertEqual(set(book["needs"]), {"authorize", "publish"})
        self.assertEqual(book["if"], "needs.authorize.outputs.publish == 'true' && !inputs.draft")
        promote = load("promote-dev.yml")["jobs"]["book"]
        self.assertEqual(promote["uses"], "./.github/workflows/modding-book-generated.yml")
        self.assertEqual(promote["if"], "needs.promote.outputs.sdk == 'true'")

    def test_sdk_publish_exports_and_publishes_generated_book_bundle_for_exact_commit(self):
        jobs = load("modding-book-generated.yml")["jobs"]
        export = jobs["book-generated"]
        publish = jobs["book-generated-publish"]
        self.assertNotIn("if", export)
        self.assertEqual(export["permissions"], {"contents": "read"})
        self.assertNotIn("environment", export)
        checkout = next(s for s in export["steps"] if s.get("uses", "").startswith("actions/checkout"))
        self.assertEqual(checkout["with"]["ref"], "${{ inputs.commit }}")
        upload = next(s for s in export["steps"] if s.get("uses", "").startswith("actions/upload-artifact"))
        self.assertNotIn("retention-days", upload["with"])
        self.assertEqual(publish["permissions"], {"contents": "write"})
        self.assertNotIn("environment", publish)
        self.assertEqual(publish["needs"], ["book-generated"])
        for step in publish["steps"]:
            self.assertNotRegex(step.get("run", ""), r"(^|\s)cargo\s")


if __name__ == "__main__":
    unittest.main()
