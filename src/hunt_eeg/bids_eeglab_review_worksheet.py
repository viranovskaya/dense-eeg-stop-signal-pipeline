"""Build a local mutable worksheet from immutable EEG review evidence."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import secrets
import stat
from pathlib import Path

from .provenance import sha256_file, verify_provenance

CHANNEL_COLUMNS = (
    "channel",
    "candidate_reason",
    "decision",
    "reviewer",
    "reviewed_at",
    "evidence",
    "notes",
)
SEGMENT_COLUMNS = (
    "candidate_id",
    "channel",
    "prompt_onset_s",
    "prompt_duration_s",
    "candidate_reason",
    "decision",
    "refined_onset_s",
    "refined_duration_s",
    "scope",
    "reviewer",
    "reviewed_at",
    "evidence",
    "notes",
)


def _capture_regular_file(path: Path) -> tuple[bytes, str]:
    _reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Worksheet input must be a regular file: {path.name}")
    data = path.read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def _read_csv_bytes(data: bytes, *, delimiter: str) -> list[dict[str, str]]:
    with io.StringIO(data.decode("utf-8"), newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream, delimiter=delimiter)]


def _verified_pack_bytes(root: Path, provenance: dict, relative: str) -> bytes:
    records = {record["path"]: record for record in provenance.get("outputs", [])}
    if relative not in records:
        raise ValueError(f"Review pack provenance does not declare {relative}")
    path = root / relative
    _reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Review pack artifact is missing or unsafe: {relative}")
    data = path.read_bytes()
    record = records[relative]
    if len(data) != record["size_bytes"] or hashlib.sha256(data).hexdigest() != record[
        "sha256"
    ]:
        raise ValueError(f"Review pack artifact changed during capture: {relative}")
    return data


def _reject_symlink_components(path: Path) -> None:
    absolute = Path(path).expanduser().absolute()
    aliases = ((Path("/var"), Path("/private/var")), (Path("/tmp"), Path("/private/tmp")))
    for alias, expected in aliases:
        if absolute.parts[:2] != alias.parts:
            continue
        if alias.is_symlink() and alias.resolve(strict=True) == expected:
            absolute = Path(expected, *absolute.parts[2:])
        break
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Path ancestry must not contain symlinks: {path}")


def _require_columns(
    rows: list[dict[str, str]], required: tuple[str, ...], label: str
) -> None:
    if not rows:
        raise ValueError(f"{label} must contain at least one prompt")
    if tuple(rows[0]) != required:
        raise ValueError(f"{label} columns do not match the controlled template")
    if any(tuple(row) != required for row in rows):
        raise ValueError(f"{label} rows do not share one exact schema")


def _relative_evidence(output: Path, review_pack: Path, relative: str) -> str:
    candidate = review_pack / relative
    _reject_symlink_components(candidate)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"Review evidence is missing or unsafe: {relative}")
    resolved_pack = review_pack.resolve(strict=True)
    resolved_candidate = candidate.resolve(strict=True)
    if not resolved_candidate.is_relative_to(resolved_pack):
        raise ValueError(f"Review evidence escapes the verified package: {relative}")
    return Path(os.path.relpath(resolved_candidate, output.parent.resolve())).as_posix()


def _bind_channel_rows(
    rows: list[dict[str, str]], index: list[dict[str, str]], output: Path, pack: Path
) -> list[dict[str, str]]:
    prompted = [row for row in rows if row["candidate_reason"] != "manual_addition"]
    additions = [row for row in rows if row["candidate_reason"] == "manual_addition"]
    if len(prompted) != len(index):
        raise ValueError("Channel worksheet does not cover every review prompt")
    if len({row["channel"] for row in rows}) != len(rows):
        raise ValueError("Channel worksheet prompt identities must be unique")
    bound = []
    for row, prompt in zip(prompted, index, strict=True):
        for field in ("channel", "candidate_reason"):
            if row[field] != prompt[field]:
                raise ValueError(f"Channel worksheet changed immutable {field}")
        bound.append(
            {
                **row,
                "figure_path": _relative_evidence(
                    output, pack, prompt["figure_path"]
                ),
            }
        )
    bound.extend({**row, "figure_path": ""} for row in additions)
    return bound


def _bind_segment_rows(
    rows: list[dict[str, str]], index: list[dict[str, str]], output: Path, pack: Path
) -> list[dict[str, str]]:
    prompted = [row for row in rows if row["candidate_reason"] != "manual_addition"]
    additions = [row for row in rows if row["candidate_reason"] == "manual_addition"]
    if len(prompted) != len(index):
        raise ValueError("Segment worksheet does not cover every review prompt")
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("Segment worksheet prompt identities must be unique")
    bound = []
    immutable = (
        "candidate_id",
        "channel",
        "prompt_onset_s",
        "prompt_duration_s",
        "candidate_reason",
    )
    for row, prompt in zip(prompted, index, strict=True):
        for field in immutable:
            if row[field] != prompt[field]:
                raise ValueError(f"Segment worksheet changed immutable {field}")
        bound.append(
            {
                **row,
                "figure_path": _relative_evidence(
                    output, pack, prompt["figure_path"]
                ),
            }
        )
    for row in additions:
        if not row["candidate_id"].startswith("manual-"):
            raise ValueError("Manual segment IDs must start with manual-")
        if row["prompt_onset_s"] or row["prompt_duration_s"]:
            raise ValueError("Manual segments must not contain QC prompt timing")
        bound.append({**row, "figure_path": ""})
    return bound


def _encoded_json(payload: object) -> str:
    data = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode()
    return base64.b64encode(data).decode("ascii")


WORKSHEET_LOGIC_JS = r"""
function rowErrors(row, kind, recordingDuration=Infinity) {
  const errors = [];
  const decision = String(row.decision || 'pending');
  const reviewer = String(row.reviewer || '').trim();
  const reviewedAt = String(row.reviewed_at || '');
  const evidence = String(row.evidence || '').trim();
  const notes = String(row.notes || '').trim();
  if (decision === 'pending') errors.push('Decision is still pending.');
  if (!reviewer) errors.push('Reviewer is required.');
  if (!/^\d{4}-\d{2}-\d{2}$/.test(reviewedAt)) errors.push('Review date is required in YYYY-MM-DD format.');
  if (!evidence) errors.push('Inspected evidence is required.');
  if (kind === 'channels') {
    if (!['pending', 'keep', 'interpolate'].includes(decision)) errors.push('Invalid channel decision.');
    if (!String(row.channel || '').trim()) errors.push('Channel is required.');
    if ((decision === 'interpolate' || row.candidate_reason === 'manual_addition') && !notes) {
      errors.push('Interpolation and manual additions require a rationale.');
    }
    return errors;
  }
  if (!['pending', 'keep', 'exclude'].includes(decision)) errors.push('Invalid segment decision.');
  const manual = row.candidate_reason === 'manual_addition';
  if (manual && !String(row.candidate_id || '').startsWith('manual-')) errors.push('Manual segment ID must start with manual-.');
  if (manual && (String(row.prompt_onset_s || '') || String(row.prompt_duration_s || ''))) errors.push('Manual additions must leave prompt timing empty.');
  if (!String(row.channel || '').trim()) errors.push('Evidence channel or all is required.');
  if (decision === 'keep') {
    if (String(row.refined_onset_s || '') || String(row.refined_duration_s || '') || String(row.scope || '')) {
      errors.push('Kept prompts must not define an exclusion.');
    }
    if (manual) errors.push('Manual segment additions must define an exclusion.');
    return errors;
  }
  if (decision === 'exclude') {
    const onsetText = String(row.refined_onset_s || '').trim();
    const durationText = String(row.refined_duration_s || '').trim();
    const onset = Number(onsetText), duration = Number(durationText);
    if (!['ica', 'epochs', 'both'].includes(String(row.scope || ''))) errors.push('Scope is required.');
    if (!onsetText || !Number.isFinite(onset) || onset < 0) errors.push('Refined onset must be finite and non-negative.');
    if (!durationText || !Number.isFinite(duration) || duration <= 0) errors.push('Refined duration must be positive.');
    if (Number.isFinite(onset) && Number.isFinite(duration) && onset + duration > Number(recordingDuration) + 1e-9) errors.push('Refined exclusion exceeds the recording duration.');
    if (!notes) errors.push('Exclusions require a rationale.');
    if (!manual && Number.isFinite(onset) && Number.isFinite(duration) && duration > 0) {
      const promptOnset = Number(row.prompt_onset_s), promptDuration = Number(row.prompt_duration_s);
      if (!Number.isFinite(promptOnset) || !Number.isFinite(promptDuration) ||
          Math.min(onset + duration, promptOnset + promptDuration) <= Math.max(onset, promptOnset)) {
        errors.push('Refined exclusion must overlap its QC prompt.');
      }
    }
  }
  return errors;
}
function rowComplete(row, kind, recordingDuration=Infinity) { return rowErrors(row, kind, recordingDuration).length === 0; }
function tsv(columns, data) {
  const clean = value => String(value ?? '').replace(/\t/g, ' ').replace(/[\r\n]+/g, ' ');
  return [columns.join('\t'), ...data.map(row => columns.map(column => clean(row[column])).join('\t'))].join('\n') + '\n';
}
"""


def _render_worksheet(payload: dict) -> str:
    encoded = _encoded_json(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' file: data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'">
<title>Dense EEG local review worksheet</title>
<style>
:root{{--ink:#173047;--muted:#5f7383;--blue:#287a8d;--blue2:#e5f2f5;--amber:#c88200;--amber2:#fff4d7;--line:#d7e1e7;--bg:#f4f7f9}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 system-ui,-apple-system,sans-serif}}
header{{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid var(--line);padding:16px 24px}}
header h1{{font-size:22px;margin:0 0 4px}} header p{{margin:0;color:var(--muted)}}
.bar{{height:8px;background:#e8eef1;border-radius:99px;overflow:hidden;margin-top:12px}} .bar span{{display:block;height:100%;background:var(--blue);width:0}}
main{{max-width:1500px;margin:auto;padding:22px;display:grid;grid-template-columns:280px minmax(0,1fr);gap:20px}}
aside,.panel{{background:#fff;border:1px solid var(--line);border-radius:14px;box-shadow:0 2px 8px #10203010}}
aside{{padding:16px;align-self:start;position:sticky;top:112px;max-height:calc(100vh - 132px);overflow:auto}}
.warning{{background:var(--amber2);border-left:5px solid var(--amber);padding:12px;margin:0 0 16px}}
label{{display:block;font-weight:650;margin:11px 0 5px}} input,select,textarea{{width:100%;font:inherit;padding:9px 10px;border:1px solid #aec0ca;border-radius:8px;background:#fff}}
textarea{{min-height:84px;resize:vertical}} button{{font:inherit;border:0;border-radius:9px;padding:9px 12px;cursor:pointer}}
.primary{{background:var(--blue);color:#fff}} .secondary{{background:#e8eef2;color:var(--ink)}} .danger{{background:#fff0f0;color:#a22}}
.tabs,.actions{{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}} .tabs button.active{{background:var(--blue);color:#fff}}
.prompt-list{{display:grid;gap:6px}} .prompt-list button{{text-align:left;background:#f0f5f7;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.prompt-list button.active{{outline:2px solid var(--blue);background:var(--blue2)}} .prompt-list button.done::after{{content:' ✓';color:#18734b}}
.panel{{padding:20px}} .meta{{display:flex;gap:8px;flex-wrap:wrap;color:var(--muted)}} .chip{{background:#edf3f6;border-radius:99px;padding:4px 9px}}
.evidence{{display:block;margin:18px 0}} .evidence img{{display:block;max-width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#fff}}
.form-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 14px}} .wide{{grid-column:1/-1}}
.status{{font-weight:700}} .hidden{{display:none!important}} .small{{font-size:13px;color:var(--muted)}}
.storage-warning{{background:#fff0f0;border-left:5px solid #b33;padding:10px;margin:10px 0}}
@media(max-width:850px){{main{{grid-template-columns:1fr}} aside{{position:static;max-height:none}} .form-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header><h1>Dense EEG local review worksheet</h1><p id="identityText"></p><p id="progressText"></p><div class="bar"><span id="progressBar"></span></div></header>
<main>
<aside>
<p class="warning"><strong>Mutable working copy.</strong><br>This page makes no decision automatically. Browser state is not provenance evidence; only successfully finalized TSV files become bound.</p>
<p id="storageWarning" class="storage-warning hidden"><strong>Browser storage is unavailable.</strong><br>Progress remains only in this open page. Export both TSV files frequently into the controlled workspace.</p>
<label for="defaultReviewer">Reviewer</label><input id="defaultReviewer" autocomplete="name">
<label for="defaultDate">Review date</label><input id="defaultDate" type="date">
<div class="tabs"><button id="channelTab">Channels</button><button id="segmentTab">Segments</button></div>
<div class="actions"><button id="pendingOnly" class="secondary">Next incomplete</button><button id="addChannel" class="secondary">Add manual channel</button><button id="addSegment" class="secondary">Add manual segment</button></div>
<div id="promptList" class="prompt-list"></div>
<hr>
<div class="actions"><button id="exportChannels" class="primary">Export channel TSV</button><button id="exportSegments" class="primary">Export segment TSV</button></div>
<p class="small">Exports preserve the controlled column order. Run the finalizer afterward; a download is not a verified decision bundle.</p>
<button id="resetState" class="danger">Reset browser state</button>
</aside>
<section class="panel">
<div id="emptyState">Select a prompt.</div>
<div id="editor" class="hidden">
<h2 id="title"></h2><div id="meta" class="meta"></div>
<a id="evidenceLink" class="evidence" target="_blank" rel="noopener"><img id="evidenceImage" loading="lazy" alt="Review evidence"></a><p id="manualEvidence" class="warning hidden">Manual addition: inspect the raw viewer and relevant global evidence, then record the exact evidence reference below.</p>
<div class="form-grid">
<div id="manualChannelWrap" class="hidden"><label for="manualChannel">Evidence channel (or all)</label><input id="manualChannel"></div>
<div><label for="decision">Decision</label><select id="decision"></select></div>
<div id="scopeWrap" class="hidden"><label for="scope">Scope</label><select id="scope"><option value=""></option><option value="ica">ica</option><option value="epochs">epochs</option><option value="both">both</option></select></div>
<div id="onsetWrap" class="hidden"><label for="onset">Refined onset (s)</label><input id="onset" inputmode="decimal"></div>
<div id="durationWrap" class="hidden"><label for="duration">Refined duration (s)</label><input id="duration" inputmode="decimal"></div>
<div><label for="reviewer">Reviewer</label><input id="reviewer"></div>
<div><label for="reviewedAt">Reviewed at</label><input id="reviewedAt" type="date"></div>
<div class="wide"><label for="evidence">Inspected evidence</label><input id="evidence" placeholder="panel, raw-viewer interval, neighbours, PSD…"></div>
<div class="wide"><label for="notes">Rationale / notes</label><textarea id="notes"></textarea></div>
</div>
<p id="validation" class="status"></p>
<div class="actions"><button id="previous" class="secondary">Previous</button><button id="next" class="primary">Save and next</button><button id="removeManual" class="danger hidden">Remove manual addition</button></div>
</div>
</section>
</main>
<script>
const initial=JSON.parse(atob('{encoded}'));
{WORKSHEET_LOGIC_JS}
const key=initial.storage_key;
let storageAvailable=false;
try{{const probe=key+':probe';localStorage.setItem(probe,'ok');storageAvailable=localStorage.getItem(probe)==='ok';localStorage.removeItem(probe)}}catch(_error){{storageAvailable=false}}
let state=null; if(storageAvailable){{try{{state=JSON.parse(localStorage.getItem(key))}}catch(_error){{state=null}}}}
if(!state||!Array.isArray(state.channels)||!Array.isArray(state.segments)) state={{channels:initial.channels,segments:initial.segments,defaults:{{reviewer:'',date:''}}}};
let mode='channels', index=0;
const $=id=>document.getElementById(id);
const fields=['decision','scope','onset','duration','reviewer','reviewedAt','evidence','notes'];
function rows(){{return state[mode]}}
function save(){{if(storageAvailable){{try{{localStorage.setItem(key,JSON.stringify(state))}}catch(_error){{storageAvailable=false;$('storageWarning').classList.remove('hidden')}}}}updateProgress()}}
function complete(row){{return rowComplete(row,mode,initial.recording_duration_s)}}
function counts(kind){{const data=state[kind],done=data.filter(row=>rowComplete(row,kind,initial.recording_duration_s)).length;return {{done,total:data.length,incomplete:data.length-done}}}}
function updateProgress(){{const c=counts('channels'),s=counts('segments'),total=c.total+s.total,done=c.done+s.done;$('progressText').textContent=`${{done}} / ${{total}} rows structurally complete · ${{initial.pack_label}}`;$('progressBar').style.width=`${{total?100*done/total:0}}%`;$('exportChannels').textContent=`Export channel TSV (${{c.incomplete}} incomplete)`;$('exportSegments').textContent=`Export segment TSV (${{s.incomplete}} incomplete)`}}
function setMode(next){{mode=next;index=0;$('channelTab').className=mode==='channels'?'active':'';$('segmentTab').className=mode==='segments'?'active':'';renderList();render()}}
function renderList(){{$('promptList').textContent='';rows().forEach((row,i)=>{{const b=document.createElement('button');b.textContent=mode==='channels'?`${{row.channel}} · ${{row.candidate_reason}}`:`${{row.candidate_id}} · ${{row.channel}}`;if(i===index)b.classList.add('active');if(complete(row))b.classList.add('done');b.onclick=()=>{{capture();index=i;renderList();render()}};$('promptList').appendChild(b)}})}}
function decisionOptions(){{return mode==='channels'?['pending','keep','interpolate']:['pending','keep','exclude']}}
function render(){{const row=rows()[index];if(!row){{$('emptyState').classList.remove('hidden');$('editor').classList.add('hidden');return}}const manual=row.candidate_reason==='manual_addition';$('emptyState').classList.add('hidden');$('editor').classList.remove('hidden');$('title').textContent=mode==='channels'?`Channel ${{row.channel||'(enter channel)'}}`:`${{row.candidate_id}} · ${{row.channel||'all'}}`;$('meta').textContent='';const values=mode==='channels'?[row.candidate_reason]:[manual?'manual interval':`${{row.prompt_onset_s}}–${{Number(row.prompt_onset_s)+Number(row.prompt_duration_s)}} s`,row.candidate_reason,'global all-channel mask if excluded'];values.forEach(value=>{{const span=document.createElement('span');span.className='chip';span.textContent=value;$('meta').appendChild(span)}});$('evidenceLink').classList.toggle('hidden',manual);$('manualEvidence').classList.toggle('hidden',!manual);if(!manual){{$('evidenceLink').href=row.figure_path;$('evidenceImage').src=row.figure_path}};$('manualChannelWrap').classList.toggle('hidden',!manual);$('manualChannel').value=row.channel||'';$('removeManual').classList.toggle('hidden',!manual);$('decision').textContent='';decisionOptions().forEach(value=>{{const option=document.createElement('option');option.value=value;option.textContent=value;$('decision').appendChild(option)}});$('decision').value=row.decision||'pending';$('scope').value=row.scope||'';$('onset').value=row.refined_onset_s||'';$('duration').value=row.refined_duration_s||'';$('reviewer').value=row.reviewer||state.defaults.reviewer||'';$('reviewedAt').value=row.reviewed_at||state.defaults.date||'';$('evidence').value=row.evidence||'';$('notes').value=row.notes||'';toggleSegmentFields();validate();}}
function toggleSegmentFields(){{const show=mode==='segments'&&$('decision').value==='exclude';['scopeWrap','onsetWrap','durationWrap'].forEach(id=>$(id).classList.toggle('hidden',!show))}}
function capture(){{const row=rows()[index];if(!row)return;if(row.candidate_reason==='manual_addition')row.channel=$('manualChannel').value.trim();row.decision=$('decision').value;row.reviewer=$('reviewer').value.trim();row.reviewed_at=$('reviewedAt').value;row.evidence=$('evidence').value.trim();row.notes=$('notes').value.trim();if(mode==='segments'){{row.refined_onset_s=$('onset').value.trim();row.refined_duration_s=$('duration').value.trim();row.scope=$('scope').value;if(row.decision==='keep'){{row.refined_onset_s='';row.refined_duration_s='';row.scope=''}}}}save();}}
function currentDraft(){{const row={{...rows()[index]}};if(row.candidate_reason==='manual_addition')row.channel=$('manualChannel').value.trim();row.decision=$('decision').value;row.reviewer=$('reviewer').value.trim();row.reviewed_at=$('reviewedAt').value;row.evidence=$('evidence').value.trim();row.notes=$('notes').value.trim();if(mode==='segments'){{row.refined_onset_s=$('onset').value.trim();row.refined_duration_s=$('duration').value.trim();row.scope=$('scope').value;if(row.decision==='keep'){{row.refined_onset_s='';row.refined_duration_s='';row.scope=''}}}}return row}}
function validate(){{const errors=rowErrors(currentDraft(),mode,initial.recording_duration_s);$('validation').textContent=errors.length?errors.join(' '):'Structurally complete for TSV export. Raw-viewer confirmation and finalizer verification remain required.';$('validation').style.color=errors.length?'#a14200':'#18734b';return !errors.length}}
function move(delta){{capture();index=Math.max(0,Math.min(rows().length-1,index+delta));renderList();render();$('promptList').children[index]?.scrollIntoView({{block:'nearest'}})}}
function download(name,columns,data){{const blob=new Blob([tsv(columns,data)],{{type:'text/tab-separated-values;charset=utf-8'}}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.hidden=true;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000)}}
fields.forEach(id=>$(id).addEventListener(id==='decision'||id==='scope'?'change':'input',()=>{{toggleSegmentFields();capture();validate();renderList()}}));
$('manualChannel').addEventListener('input',()=>{{capture();validate();renderList()}});
$('defaultReviewer').value=state.defaults.reviewer||'';$('defaultDate').value=state.defaults.date||'';
$('defaultReviewer').oninput=()=>{{state.defaults.reviewer=$('defaultReviewer').value.trim();if(!$('reviewer').value.trim())$('reviewer').value=state.defaults.reviewer;capture();validate();renderList()}};$('defaultDate').oninput=()=>{{state.defaults.date=$('defaultDate').value;if(!$('reviewedAt').value)$('reviewedAt').value=state.defaults.date;capture();validate();renderList()}};
$('channelTab').onclick=()=>{{capture();setMode('channels')}};$('segmentTab').onclick=()=>{{capture();setMode('segments')}};$('previous').onclick=()=>move(-1);$('next').onclick=()=>move(1);
$('pendingOnly').onclick=()=>{{capture();const next=rows().findIndex((row,i)=>i>index&&!complete(row));index=next>=0?next:Math.max(0,rows().findIndex(row=>!complete(row)));renderList();render()}};
$('addChannel').onclick=()=>{{capture();const row={{channel:'',candidate_reason:'manual_addition',decision:'pending',reviewer:state.defaults.reviewer||'',reviewed_at:state.defaults.date||'',evidence:'',notes:'',figure_path:''}};state.channels.push(row);setMode('channels');index=state.channels.length-1;save();renderList();render()}};
$('addSegment').onclick=()=>{{capture();const used=new Set(state.segments.map(row=>row.candidate_id));let n=1,id;do{{id=`manual-${{String(n++).padStart(4,'0')}}`}}while(used.has(id));const row={{candidate_id:id,channel:'all',prompt_onset_s:'',prompt_duration_s:'',candidate_reason:'manual_addition',decision:'exclude',refined_onset_s:'',refined_duration_s:'',scope:'',reviewer:state.defaults.reviewer||'',reviewed_at:state.defaults.date||'',evidence:'',notes:'',figure_path:''}};state.segments.push(row);setMode('segments');index=state.segments.length-1;save();renderList();render()}};
$('removeManual').onclick=()=>{{const row=rows()[index];if(row?.candidate_reason!=='manual_addition')return;if(confirm('Remove this manual addition from the mutable worksheet?')){{rows().splice(index,1);index=Math.max(0,Math.min(index,rows().length-1));save();renderList();render()}}}};
$('exportChannels').onclick=()=>{{capture();download('channel_decisions.tsv',initial.channel_columns,state.channels)}};$('exportSegments').onclick=()=>{{capture();download('segment_decisions.tsv',initial.segment_columns,state.segments)}};
$('resetState').onclick=()=>{{if(confirm('Delete browser-saved decisions for this exact worksheet?')){{if(storageAvailable){{try{{localStorage.removeItem(key)}}catch(_error){{}}}}location.reload()}}}};
if(!storageAvailable)$('storageWarning').classList.remove('hidden');$('identityText').textContent=`Session ${{initial.worksheet_session_id}} · QC ${{initial.qc_short_id}} · pack ${{initial.pack_short_id}}`;setMode('channels');updateProgress();
</script>
</body></html>"""


def build_review_worksheet(
    review_pack: Path,
    qc_root: Path,
    channel_decisions: Path,
    segment_decisions: Path,
    output: Path,
    session_id: str,
) -> Path:
    """Create one local HTML worksheet without changing evidence or decisions."""
    review_pack = Path(review_pack).expanduser().absolute()
    qc_root = Path(qc_root).expanduser().absolute()
    channel_decisions = Path(channel_decisions).expanduser().absolute()
    segment_decisions = Path(segment_decisions).expanduser().absolute()
    session_id = str(session_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", session_id):
        raise ValueError(
            "Worksheet session ID must use 1-64 letters, digits, dot, underscore or hyphen"
        )
    raw_output = Path(output).expanduser()
    if ".." in raw_output.parts:
        raise ValueError("Worksheet output must not contain parent traversal")
    output = raw_output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Worksheet output already exists: {output}")
    _reject_symlink_components(review_pack)
    _reject_symlink_components(qc_root)
    _reject_symlink_components(output.parent)
    resolved_pack = review_pack.resolve(strict=True)
    resolved_qc = qc_root.resolve(strict=True)
    ancestor = output.parent
    while not ancestor.exists() and not ancestor.is_symlink():
        if ancestor == ancestor.parent:
            break
        ancestor = ancestor.parent
    if ancestor.is_symlink():
        raise ValueError("Worksheet output ancestry must not contain a symlink")
    resolved_ancestor = ancestor.resolve(strict=True)
    if any(
        resolved_ancestor.is_relative_to(root)
        for root in (resolved_pack, resolved_qc)
    ):
        raise ValueError("Mutable worksheet must remain outside immutable packages")
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output.parent)
    if any(
        output.parent.resolve(strict=True).is_relative_to(root)
        for root in (resolved_pack, resolved_qc)
    ):
        raise ValueError("Mutable worksheet must remain outside immutable packages")
    output = output.parent.resolve(strict=True) / output.name
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Worksheet output already exists: {output}")

    provenance = verify_provenance(review_pack)
    qc_provenance = verify_provenance(qc_root)
    qc_identity = provenance.get("qc", {})
    if (
        qc_identity.get("provenance_sha256")
        != sha256_file(qc_root / "provenance.json")
        or qc_identity.get("core_sha256") != qc_provenance.get("core_sha256")
        or qc_identity.get("summary_sha256")
        != sha256_file(qc_root / "qc_summary.json")
    ):
        raise ValueError("Review pack does not match the supplied QC package")
    qc_summary = json.loads(
        _verified_pack_bytes(qc_root, qc_provenance, "qc_summary.json").decode(
            "utf-8"
        )
    )
    recording_duration_s = float(qc_summary.get("recording_duration_s", 0.0))
    if not math.isfinite(recording_duration_s) or recording_duration_s <= 0:
        raise ValueError("QC recording duration must be finite and positive")
    pack_provenance_sha256 = sha256_file(review_pack / "provenance.json")
    summary = json.loads(
        _verified_pack_bytes(
            review_pack, provenance, "review_pack_summary.json"
        ).decode("utf-8")
    )
    if summary.get("publication_allowed") is not False:
        raise ValueError("Review worksheet requires controlled non-public evidence")
    channel_index = _read_csv_bytes(
        _verified_pack_bytes(
            review_pack, provenance, "channel_figure_index.csv"
        ),
        delimiter=",",
    )
    segment_index = _read_csv_bytes(
        _verified_pack_bytes(
            review_pack, provenance, "segment_figure_index.csv"
        ),
        delimiter=",",
    )
    channel_bytes, channel_sha256 = _capture_regular_file(channel_decisions)
    segment_bytes, segment_sha256 = _capture_regular_file(segment_decisions)
    channel_rows = _read_csv_bytes(channel_bytes, delimiter="\t")
    segment_rows = _read_csv_bytes(segment_bytes, delimiter="\t")
    _require_columns(channel_rows, CHANNEL_COLUMNS, "Channel worksheet")
    _require_columns(segment_rows, SEGMENT_COLUMNS, "Segment worksheet")

    payload = {
        "pack_label": "controlled evidence; publication not allowed",
        "pack_provenance_sha256": pack_provenance_sha256,
        "pack_core_sha256": provenance["core_sha256"],
        "qc_provenance_sha256": qc_identity["provenance_sha256"],
        "worksheet_session_id": session_id,
        "pack_short_id": pack_provenance_sha256[:12],
        "qc_short_id": qc_identity["provenance_sha256"][:12],
        "recording_duration_s": recording_duration_s,
        "channel_columns": CHANNEL_COLUMNS,
        "segment_columns": SEGMENT_COLUMNS,
        "channels": _bind_channel_rows(
            channel_rows, channel_index, output, review_pack
        ),
        "segments": _bind_segment_rows(
            segment_rows, segment_index, output, review_pack
        ),
    }
    identity = "|".join(
        (
            payload["pack_provenance_sha256"],
            channel_sha256,
            segment_sha256,
            session_id,
        )
    )
    payload["storage_key"] = "dense-eeg-review:" + hashlib.sha256(
        identity.encode()
    ).hexdigest()
    rendered = _render_worksheet(payload)
    rendered_sha256 = hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    def verify_context() -> None:
        final_provenance = verify_provenance(review_pack)
        final_qc_provenance = verify_provenance(qc_root)
        if (
            sha256_file(review_pack / "provenance.json")
            != pack_provenance_sha256
            or final_provenance["core_sha256"] != provenance["core_sha256"]
        ):
            raise ValueError("Review pack changed before worksheet publication")
        if (
            sha256_file(qc_root / "provenance.json")
            != qc_identity["provenance_sha256"]
            or final_qc_provenance["core_sha256"] != qc_identity["core_sha256"]
            or sha256_file(qc_root / "qc_summary.json")
            != qc_identity["summary_sha256"]
        ):
            raise ValueError("QC package changed before worksheet publication")
        _, current_channel_sha256 = _capture_regular_file(channel_decisions)
        _, current_segment_sha256 = _capture_regular_file(segment_decisions)
        if (
            current_channel_sha256 != channel_sha256
            or current_segment_sha256 != segment_sha256
        ):
            raise ValueError("Decision working copies changed during worksheet build")

    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        parent_flags |= os.O_NOFOLLOW
    parent_fd = os.open(output.parent, parent_flags)
    parent_identity = (os.fstat(parent_fd).st_dev, os.fstat(parent_fd).st_ino)
    temporary_name = f".{output.name}.{secrets.token_hex(12)}"
    published_identity: tuple[int, int] | None = None

    def verify_parent_identity() -> None:
        _reject_symlink_components(output.parent)
        current = output.parent.stat()
        if (
            not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != parent_identity
        ):
            raise ValueError("Worksheet output parent changed during publication")

    def sha256_at(name: str) -> str:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(name, flags, dir_fd=parent_fd)
        try:
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            return digest.hexdigest()
        finally:
            os.close(descriptor)

    try:
        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            create_flags |= os.O_NOFOLLOW
        descriptor = os.open(
            temporary_name, create_flags, 0o600, dir_fd=parent_fd
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        verify_parent_identity()
        verify_context()
        try:
            os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"Worksheet output already exists: {output}")
        verify_parent_identity()
        os.link(
            temporary_name,
            output.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        published = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        published_identity = (published.st_dev, published.st_ino)
        if (
            not stat.S_ISREG(published.st_mode)
            or stat.S_IMODE(published.st_mode) != 0o600
            or sha256_at(output.name) != rendered_sha256
        ):
            raise ValueError("Published worksheet failed its final integrity check")
        verify_parent_identity()
        verify_context()
        return output
    except BaseException:
        if published_identity is not None:
            try:
                current = os.stat(
                    output.name, dir_fd=parent_fd, follow_symlinks=False
                )
                if (current.st_dev, current.st_ino) == published_identity:
                    os.unlink(output.name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)
