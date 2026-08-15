from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from hunt_eeg import bids_eeglab_review_worksheet
from hunt_eeg.bids_eeglab_review_worksheet import (
    WORKSHEET_LOGIC_JS,
    build_review_worksheet,
)
from hunt_eeg.bids_eeglab import publish_inventory
from hunt_eeg.bids_eeglab_qc import publish_qc
from hunt_eeg.bids_eeglab_review_pack import build_review_pack
from test_bids_eeglab import _fixture


class ReviewWorksheetTests(unittest.TestCase):
    @staticmethod
    def _build(pack, qc, channel, segment, output):
        return build_review_worksheet(
            pack, qc, channel, segment, output, "test-reviewer-a"
        )

    def _pack_and_templates(self, root: Path):
        inputs, profile = _fixture(root)
        payload = json.loads(profile.read_text(encoding="utf-8"))
        payload["expected"]["power_line_frequency_hz"] = 40.0
        payload["qc"] = {
            "screen_filter_hz": [1.0, 40.0],
            "full_recording_window_seconds": 2.0,
            "robust_z_threshold": 0.5,
            "flat_fraction_threshold": 0.1,
            "line_noise_ratio_db_threshold": 20.0,
        }
        profile.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        eeg_json = inputs.dataset_root / "task-contrast_eeg.json"
        eeg_payload = json.loads(eeg_json.read_text(encoding="utf-8"))
        eeg_payload["PowerLineFrequency"] = 40.0
        eeg_json.write_text(
            json.dumps(eeg_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        inventory = publish_inventory(inputs, profile, root / "inventory")
        qc = publish_qc(inputs, profile, inventory, root / "qc")
        pack = build_review_pack(inputs, profile, qc, root / "pack")
        channel = root / "channel.tsv"
        segment = root / "segment.tsv"
        channel.write_bytes((qc / "channel_review_template.tsv").read_bytes())
        segment.write_bytes((qc / "segment_review_template.tsv").read_bytes())
        return pack, qc, channel, segment

    @staticmethod
    def _payload(rendered: str) -> dict:
        match = re.search(r"JSON\.parse\(atob\('([^']+)'\)\)", rendered)
        if match is None:
            raise AssertionError("Worksheet payload was not found")
        return json.loads(base64.b64decode(match.group(1)))

    def test_worksheet_is_local_mutable_and_covers_every_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack, qc, channel, segment = self._pack_and_templates(root)
            output = self._build(
                pack, qc, channel, segment, root / "workspace" / "review.html"
            )
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("Mutable working copy", rendered)
            self.assertIn("makes no decision automatically", rendered)
            self.assertIn("Export channel TSV", rendered)
            self.assertIn("Export segment TSV", rendered)
            self.assertIn("localStorage", rendered)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertFalse((pack / "review.html").exists())
            payload = self._payload(rendered)
            self.assertEqual(payload["worksheet_session_id"], "test-reviewer-a")
            self.assertIn("id=\"identityText\"", rendered)
            self.assertEqual(len(payload["channels"]), 2)
            self.assertGreater(len(payload["segments"]), 0)
            for row in payload["channels"] + payload["segments"]:
                self.assertTrue((output.parent / row["figure_path"]).is_file())

    def test_changed_prompt_identity_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack, qc, channel, segment = self._pack_and_templates(root)
            table = pd.read_csv(segment, sep="\t", dtype=str, keep_default_na=False)
            table.loc[0, "prompt_onset_s"] = "999"
            table.to_csv(segment, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "immutable prompt_onset_s"):
                self._build(pack, qc, channel, segment, root / "review.html")
            output = root / "exists.html"
            output.write_text("do not overwrite", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self._build(pack, qc, channel, segment, output)

    def test_worksheet_cannot_be_written_inside_immutable_pack(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack, qc, channel, segment = self._pack_and_templates(root)
            nested = pack / "mutable" / "worksheet.html"
            with self.assertRaisesRegex(ValueError, "outside immutable packages"):
                self._build(pack, qc, channel, segment, nested)
            self.assertFalse(nested.parent.exists())

    def test_symlinked_working_copy_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack, qc, channel, segment = self._pack_and_templates(root)
            channel_link = root / "channel-link.tsv"
            channel_link.symlink_to(channel)
            with self.assertRaisesRegex(ValueError, "symlinks|regular file"):
                self._build(
                    pack, qc, channel_link, segment, root / "review.html"
                )

    def test_transient_review_pack_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack, qc, channel, segment = self._pack_and_templates(root)
            original_capture = bids_eeglab_review_worksheet._verified_pack_bytes
            changed = False

            def mutate_then_restore(pack_root, provenance, relative):
                nonlocal changed
                if relative != "channel_figure_index.csv" or changed:
                    return original_capture(pack_root, provenance, relative)
                changed = True
                path = pack_root / relative
                original = path.read_bytes()
                path.write_bytes(original.replace(b"pending", b"altered", 1))
                try:
                    return original_capture(pack_root, provenance, relative)
                finally:
                    path.write_bytes(original)

            output = root / "review.html"
            with mock.patch.object(
                bids_eeglab_review_worksheet,
                "_verified_pack_bytes",
                side_effect=mutate_then_restore,
            ):
                with self.assertRaisesRegex(ValueError, "changed during capture"):
                    self._build(pack, qc, channel, segment, output)
            self.assertFalse(output.exists())

    def test_unicode_hostile_text_and_manual_additions_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack, qc, channel, segment = self._pack_and_templates(root)
            channel_table = pd.read_csv(
                channel, sep="\t", dtype=str, keep_default_na=False
            )
            channel_table.loc[0, ["decision", "reviewer", "reviewed_at", "evidence"]] = [
                "keep",
                "Дарья",
                "2026-08-15",
                "сырой сигнал и PSD",
            ]
            channel_table.loc[0, "notes"] = "</script><img src=x onerror=alert(1)>"
            channel_table.loc[len(channel_table)] = {
                "channel": "manual channel",
                "candidate_reason": "manual_addition",
                "decision": "pending",
                "reviewer": "",
                "reviewed_at": "",
                "evidence": "",
                "notes": "",
            }
            channel_table.to_csv(channel, sep="\t", index=False)
            segment_table = pd.read_csv(
                segment, sep="\t", dtype=str, keep_default_na=False
            )
            segment_table.loc[len(segment_table)] = {
                "candidate_id": "manual-0001",
                "channel": "all",
                "prompt_onset_s": "",
                "prompt_duration_s": "",
                "candidate_reason": "manual_addition",
                "decision": "exclude",
                "refined_onset_s": "1.25",
                "refined_duration_s": "0.5",
                "scope": "both",
                "reviewer": "Дарья",
                "reviewed_at": "2026-08-15",
                "evidence": "raw viewer",
                "notes": "вне автоматического промпта",
            }
            segment_table.to_csv(segment, sep="\t", index=False)
            output = self._build(
                pack, qc, channel, segment, root / "workspace" / "review.html"
            )
            rendered = output.read_text(encoding="utf-8")
            payload = self._payload(rendered)
            self.assertEqual(payload["channels"][0]["reviewer"], "Дарья")
            self.assertEqual(payload["channels"][0]["evidence"], "сырой сигнал и PSD")
            self.assertEqual(payload["channels"][-1]["figure_path"], "")
            self.assertEqual(payload["segments"][-1]["candidate_id"], "manual-0001")
            self.assertEqual(payload["segments"][-1]["figure_path"], "")
            self.assertNotIn("</script><img src=x onerror=alert(1)>", rendered)

    def test_nested_symlink_ancestry_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack, qc, channel, segment = self._pack_and_templates(root)
            target = root / "target"
            target.mkdir()
            link = root / "linked"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                self._build(
                    pack, qc, channel, segment, link / "nested" / "review.html"
                )
            linked_channel = link / "channel.tsv"
            linked_channel.write_bytes(channel.read_bytes())
            with self.assertRaisesRegex(ValueError, "symlinks"):
                self._build(
                    pack, qc, linked_channel, segment, root / "review.html"
                )

    def test_publish_race_does_not_overwrite_and_temp_is_cleaned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack, qc, channel, segment = self._pack_and_templates(root)
            output = root / "workspace" / "review.html"

            def race(_source, _destination, **_kwargs):
                output.write_text("racing writer", encoding="utf-8")
                raise FileExistsError("destination appeared")

            with mock.patch.object(os, "link", side_effect=race):
                with self.assertRaises(FileExistsError):
                    self._build(pack, qc, channel, segment, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "racing writer")
            self.assertEqual(list(output.parent.glob(f".{output.name}.*")), [])

    def test_temp_write_failure_is_cleaned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack, qc, channel, segment = self._pack_and_templates(root)
            output = root / "workspace" / "review.html"
            with mock.patch.object(os, "fsync", side_effect=OSError("fsync failed")):
                with self.assertRaisesRegex(OSError, "fsync failed"):
                    self._build(pack, qc, channel, segment, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(output.parent.glob(f".{output.name}.*")), [])

    def test_parent_swap_during_publication_is_rejected_and_cleaned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack, qc, channel, segment = self._pack_and_templates(root)
            output = root / "workspace" / "review.html"
            output.parent.mkdir()
            original_link = os.link
            moved = root / "moved-workspace"
            replacement = root / "replacement"

            def swap_then_link(source, destination, **kwargs):
                output.parent.rename(moved)
                replacement.mkdir()
                output.parent.symlink_to(replacement, target_is_directory=True)
                return original_link(source, destination, **kwargs)

            with mock.patch.object(os, "link", side_effect=swap_then_link):
                with self.assertRaisesRegex(ValueError, "symlinks|parent changed"):
                    self._build(pack, qc, channel, segment, output)
            self.assertFalse((moved / output.name).exists())
            self.assertEqual(list(moved.glob(f".{output.name}.*")), [])

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for JS checks")
    def test_browser_logic_validates_complete_rows_and_exact_tsv(self):
        script = WORKSHEET_LOGIC_JS + r"""
const channel={channel:'E1',candidate_reason:'manual_addition',decision:'keep',reviewer:'Дарья',reviewed_at:'2026-08-15',evidence:'raw viewer',notes:'rationale'};
if(!rowComplete(channel,'channels')) process.exit(10);
const incomplete={...channel,evidence:''};
if(rowComplete(incomplete,'channels')) process.exit(11);
const segment={candidate_id:'segment-1',channel:'E1',prompt_onset_s:'10',prompt_duration_s:'2',candidate_reason:'high_range',decision:'exclude',refined_onset_s:'11',refined_duration_s:'0.5',scope:'both',reviewer:'D',reviewed_at:'2026-08-15',evidence:'panel',notes:'artifact'};
if(!rowComplete(segment,'segments',30)) process.exit(12);
const nonoverlap={...segment,refined_onset_s:'20'};
if(rowComplete(nonoverlap,'segments',30)) process.exit(13);
const beyond={...segment,refined_onset_s:'29.8',refined_duration_s:'0.5',prompt_onset_s:'29',prompt_duration_s:'1'};
if(rowComplete(beyond,'segments',30)) process.exit(15);
const actual=tsv(['reviewer','notes'],[{reviewer:'Дарья',notes:'a\tb\nc'}]);
if(actual!=='reviewer\tnotes\nДарья\ta b c\n') process.exit(14);
"""
        subprocess.run(["node", "-e", script], check=True)

    def test_session_id_separates_browser_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack, qc, channel, segment = self._pack_and_templates(root)
            first = build_review_worksheet(
                pack,
                qc,
                channel,
                segment,
                root / "first.html",
                "reviewer-a",
            )
            second = build_review_worksheet(
                pack,
                qc,
                channel,
                segment,
                root / "second.html",
                "reviewer-b",
            )
            first_payload = self._payload(first.read_text(encoding="utf-8"))
            second_payload = self._payload(second.read_text(encoding="utf-8"))
            self.assertNotEqual(
                first_payload["storage_key"], second_payload["storage_key"]
            )
            self.assertNotEqual(
                first_payload["worksheet_session_id"],
                second_payload["worksheet_session_id"],
            )

    def test_headless_browser_full_review_and_storage_fallback(self):
        node_path = os.environ.get("DENSE_EEG_PLAYWRIGHT_NODE_PATH")
        browser = os.environ.get("DENSE_EEG_BROWSER_EXECUTABLE")
        node = shutil.which("node")
        if not node_path or not browser or not node:
            self.skipTest("Optional Playwright/Chromium integration environment unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack, qc, channel, segment = self._pack_and_templates(root)
            output = self._build(
                pack, qc, channel, segment, root / "workspace" / "review.html"
            )
            script = r"""
const fs=require('fs');
const {chromium}=require('playwright');
const [htmlPath,browserPath]=process.argv.slice(1);
const html=fs.readFileSync(htmlPath,'utf8');
function assert(value,message){if(!value)throw new Error(message)}
(async()=>{
  const instance=await chromium.launch({headless:true,executablePath:browserPath});
  const context=await instance.newContext({acceptDownloads:true});
  const page=await context.newPage();
  await page.addInitScript(()=>{
    window.__worksheetDownloads=[];
    class CapturedBlob {
      constructor(parts,options){this.parts=parts;this.options=options}
    }
    window.Blob=CapturedBlob;
    URL.createObjectURL=blob=>{window.__worksheetBlob=blob;return 'blob:https://worksheet.local/captured'};
    URL.revokeObjectURL=()=>{};
    HTMLAnchorElement.prototype.click=function(){
      window.__worksheetDownloads.push({
        name:this.download,
        text:(window.__worksheetBlob?.parts||[]).join(''),
      });
    };
  });
  await page.route('**/*',route=>route.request().url()==='https://worksheet.local/'?route.fulfill({status:200,contentType:'text/html',body:html}):route.abort());
  await page.goto('https://worksheet.local/');
  assert((await page.locator('#identityText').textContent()).includes('test-reviewer-a'),'visible session missing');
  await page.fill('#defaultReviewer','Reviewer A');
  await page.fill('#defaultDate','2026-08-15');
  const firstTitle=await page.locator('#title').textContent();
  await page.selectOption('#decision','keep');
  await page.fill('#evidence','channel panel and raw viewer');
  assert((await page.locator('#progressText').textContent()).startsWith('1 /'),'full-row progress did not update');
  await page.click('#pendingOnly');
  assert((await page.locator('#title').textContent())!==firstTitle,'navigation did not advance');
  await page.click('#addChannel');
  await page.fill('#manualChannel','E99');
  await page.selectOption('#decision','keep');
  await page.fill('#evidence','raw viewer');
  await page.fill('#notes','manual finding rationale');
  await page.click('#exportChannels');
  const channelText=await page.evaluate(()=>window.__worksheetDownloads[window.__worksheetDownloads.length-1].text);
  assert(channelText.includes('E99\tmanual_addition\tkeep'),'manual channel export missing');
  await page.reload();
  assert(await page.locator('#promptList button',{hasText:'E99'}).count()===1,'localStorage restore failed');
  await page.click('#addSegment');
  await page.fill('#onset','1.5');
  await page.fill('#duration','0.25');
  await page.selectOption('#scope','both');
  await page.fill('#evidence','raw viewer interval');
  await page.fill('#notes','manual interval rationale');
  await page.click('#exportSegments');
  const segmentText=await page.evaluate(()=>window.__worksheetDownloads[window.__worksheetDownloads.length-1].text);
  assert(segmentText.includes('manual-0001\tall\t\t\tmanual_addition\texclude'),'manual segment export missing');
  await context.close();
  const noStorage=await instance.newContext();
  await noStorage.addInitScript(()=>Object.defineProperty(window,'localStorage',{configurable:true,get(){throw new Error('disabled')}}));
  const fallback=await noStorage.newPage();
  await fallback.route('**/*',route=>route.request().url()==='https://worksheet.local/'?route.fulfill({status:200,contentType:'text/html',body:html}):route.abort());
  await fallback.goto('https://worksheet.local/');
  assert(await fallback.locator('#storageWarning').isVisible(),'storage fallback warning missing');
  await noStorage.close();
  await instance.close();
})().catch(error=>{console.error(error);process.exit(1)});
"""
            environment = os.environ.copy()
            environment["NODE_PATH"] = node_path
            subprocess.run(
                [node, "-e", script, str(output), browser],
                check=True,
                env=environment,
                timeout=60,
            )


if __name__ == "__main__":
    unittest.main()
