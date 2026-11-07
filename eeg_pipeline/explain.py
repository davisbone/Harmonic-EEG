"""Regenerate the standalone visual walkthrough from the verified published table.

Run: uv run python -m eeg_pipeline.explain
"""
import base64
import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PREDICTORS, ROIS, WINDOWS
from .source import file_checksum

ROOT = Path(__file__).resolve().parent.parent
COLORS = ["#1d7782", "#b66e29", "#80629e", "#4d885e", "#c0615e", "#5c7baf", "#798743"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.labelcolor": "#354449", "text.color": "#25363d"})


def figure_uri(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def picture(uri, caption):
    return f'<figure><img src="{uri}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'


def build_explainer(destination=None):
    output = ROOT / "results"
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    for name, digest in provenance["files"].items():
        if file_checksum(output / name, "sha256") != digest:
            raise ValueError(f"Published output changed: {name}")
    if provenance["predictors"] != PREDICTORS:
        raise ValueError("Published predictor list differs from configuration")
    events = pd.read_parquet(output / "events.parquet")
    ledger = pd.read_csv(output / "event_ledger.csv.gz", encoding="utf-8", keep_default_na=False)
    audit = pd.read_csv(output / "participant_audit.csv", encoding="utf-8")
    if events.empty:
        raise ValueError("The walkthrough requires a nonempty analysis table")
    with np.load(output / "example.npz") as saved:
        example = {name: saved[name] for name in saved.files}
    n_people, n_events = events.subject.nunique(), len(events)
    deviant_pct = 100 * events.target.mean()
    selected = int(np.flatnonzero(example["included"])[0])
    event_id = str(example["event_ids"][selected])
    channel_names = list(example["channel_names"])
    times = example["times_s"]

    fig, axes = plt.subplots(2, 1, figsize=(9, 4.7), sharex=True)
    channel = channel_names.index("Fz")
    raw = example["raw_uv"][selected, channel]
    axes[0].plot(example["raw_times_s"] * 1000, raw - raw.mean(), color=COLORS[1], lw=1)
    axes[0].set(title="Original Fz voltage, centered for display", ylabel="µV")
    axes[1].plot(times * 1000, example["baseline_corrected_uv"][selected, channel], color=COLORS[0], lw=1.3)
    axes[1].set(title="After filtering, artifact correction, reference and baseline", xlabel="Time from middle chord (ms)", ylabel="µV")
    for ax in axes:
        ax.axvline(0, color="#65757b", lw=.7, ls="--")
    fig.tight_layout()
    raw_figure = picture(figure_uri(fig), f"Actual retained event {event_id}, electrode Fz. The panels have their own vertical scales. Display-centering the raw trace is not an extra cleaning step.")

    region = "FCmid"
    picks = [channel_names.index(name) for name in ROIS[region]]
    before = example["clean_uv"][selected, picks].mean(axis=0)
    signal = example["baseline_corrected_uv"][selected, picks].mean(axis=0)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(times * 1000, before, color="#a7b2b7", lw=1, label="Before baseline subtraction")
    ax.plot(times * 1000, signal, color=COLORS[0], lw=1.4, label="After baseline subtraction")
    ax.axvspan(-100, 0, color="#b9c3c8", alpha=.2)
    cells = []
    row = events.set_index("event_id").loc[event_id]
    for i, (window, (start, stop)) in enumerate(WINDOWS.items()):
        mean = signal[(times >= start) & (times < stop)].mean()
        field = f"erp_{region}_{window}_uv"
        if not np.isclose(mean, row[field], rtol=1e-9, atol=1e-9):
            raise ValueError(f"Worked example and published feature disagree: {field}")
        color = COLORS[i + 1]
        ax.axvspan(start * 1000, stop * 1000, color=color, alpha=.13)
        ax.hlines(mean, start * 1000, stop * 1000, color=color, lw=3, label=f"{window} ms: {mean:.2f} µV")
        cells.append(f'<tr><td>{start*1000:.0f}–{stop*1000:.0f} ms</td><td>{mean:.3f} µV</td></tr>')
    ax.axhline(0, color="#839095", lw=.6)
    ax.axvline(0, color="#839095", lw=.6)
    ax.set(xlabel="Time from middle chord (ms)", ylabel="Mean voltage (µV)", title=f"One event, one region, three features · {region}")
    ax.legend(ncol=2, fontsize=8); fig.tight_layout()
    feature_figure = picture(figure_uri(fig), "Horizontal colored lines are the exact means written to the table. The gray baseline interval occurs during the preceding chord, not silence.")

    eligible_participants = int(
        (ledger.boundary_eligible & (ledger.reason != "participant:too_many_bad_channels")).sum()
    )
    participant_excluded = int(ledger.boundary_eligible.sum()) - eligible_participants
    signal_excluded = eligible_participants - int(ledger.signal_pass.sum())
    min_retention = 100 * float(audit.loc[audit.n_included > 0, "retention"].min())
    counts = [
        len(ledger),
        int(ledger.source_available.sum()),
        int(ledger.boundary_eligible.sum()),
        eligible_participants,
        int(ledger.signal_pass.sum()),
        int(ledger.included.sum()),
    ]
    labels = [
        "Candidate events",
        "Available signal",
        "Outside edge guards",
        "Eligible participants (≤6 bad ch)",
        "Pass event QC",
        "Final table (≥60% retention)",
    ]
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    ax.barh(labels[::-1], counts[::-1], color=[COLORS[0], "#4e929b", "#77aeb6", "#9dc5cb", "#bed9dd", "#dce9eb"])
    for index, count in enumerate(counts[::-1]):
        ax.text(count + max(counts) * .015, index, f"{count:,}", va="center", fontsize=10)
    ax.set(xlabel="Events", xlim=(0, max(counts) * 1.18), title="Every missing row has an explanation")
    fig.tight_layout()
    flow_figure = picture(
        figure_uri(fig),
        f"Measured counts from the complete event ledger. Three participants exceeding six bad electrodes remove {participant_excluded:,} edge-eligible events together, {signal_excluded:,} events fail individual amplitude checks, and all {n_people} remaining participants exceed the 60% retention threshold (lowest: {min_retention:.1f}%).",
    )

    class_counts = events.condition.value_counts().reindex(["standard", "function_dev", "exemplar_dev"], fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(["Standard", "Function\ndeviant", "Exemplar\ndeviant"], class_counts, color=[COLORS[0], COLORS[1], COLORS[2]])
    axes[0].set(title="Retained classes", ylabel="Events")
    feature = "erp_FCmid_150_250_uv"
    for target, name, color in [(0, "Standard", COLORS[0]), (1, "Deviant", COLORS[1])]:
        values = events.loc[events.target == target, feature]
        axes[1].hist(values, bins=45, histtype="step", density=True, color=color, label=name, lw=1.3)
    axes[1].set(title="One feature does not give a clean separation", xlabel="FCmid, 150–250 ms (µV)", ylabel="Density")
    axes[1].legend(fontsize=8); fig.tight_layout()
    distribution_figure = picture(figure_uri(fig), "Actual class counts and an illustrative feature distribution. This plot is descriptive, not a cross-validated performance estimate.")

    subj_cond = events.groupby(["subject", "condition"])[PREDICTORS].mean()
    cond_means = subj_cond.groupby("condition").mean()
    cond_sems = subj_cond.groupby("condition").sem()
    window_keys = list(WINDOWS.keys())
    window_labels = [f"{int(start*1000)}–{int(stop*1000)} ms" for start, stop in WINDOWS.values()]
    x_pos = np.arange(len(window_keys))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    cond_specs = [
        ("standard", "Standard", COLORS[0], -0.15),
        ("exemplar_dev", "Exemplar deviant", COLORS[2], 0.0),
        ("function_dev", "Function deviant", COLORS[1], 0.15),
    ]
    for ax, roi, title in zip(axes, ["FCmid", "Pmid"], ["FCmid (frontocentral)", "Pmid (parietal)"]):
        cols = [f"erp_{roi}_{w}_uv" for w in window_keys]
        trial_sds = events[cols].std()
        ax.axhline(0, color="#839095", lw=.6, ls="--")
        for cond, label, color, offset in cond_specs:
            means = [cond_means.loc[cond, c] for c in cols]
            sems = [cond_sems.loc[cond, c] for c in cols]
            ax.errorbar(x_pos + offset, means, yerr=sems, fmt="o", color=color, capsize=3, lw=1.3, ms=5.5, label=label)
        ax.set_xticks(x_pos, window_labels)
        ax.set_title(f"{title}\n(single-trial SD: {trial_sds.min():.1f}–{trial_sds.max():.1f} µV)", fontsize=10.5)
        ax.set_xlabel("Summary window")
    axes[0].set_ylabel("Participant-weighted mean (µV)")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    window_avg_figure = picture(
        figure_uri(fig),
        f"Participant-weighted window averages (mean within each participant and condition, then averaged equally across the {n_people} retained participants; error bars show ±1 SEM across participants). These are three discrete window summaries, not a full ERP time course, and they do not prove stimulus alignment or artifact-free cleaning.",
    )

    source_counts = audit.source_status.value_counts()
    condition_retention = ledger.groupby("condition").agg(candidates=("event_id", "size"), retained=("included", "sum"))
    condition_retention["retained_percent"] = 100 * condition_retention.retained / condition_retention.candidates
    retention_table = condition_retention.round(1).to_html(classes="data", border=0)
    regions_table = ''.join(f'<tr><td><b>{name}</b></td><td>{", ".join(names)}</td></tr>' for name, names in ROIS.items())
    guards = audit.filter_guard_s.dropna()
    guard = float(guards.iloc[0])
    content = TEMPLATE.replace("{{PEOPLE}}", str(n_people)).replace("{{EVENTS}}", f"{n_events:,}")
    substitutions = {
        "GB": f"{provenance['full_release_bytes']/1e9:.1f}", "DEVIANT": f"{deviant_pct:.1f}",
        "SOURCE_COUNTS": f"{source_counts.get('complete',0)} complete · {source_counts.get('verified_prefix',0)} usable partial · {source_counts.get('severely_truncated',0)} severely truncated",
        "GUARD": f"{guard:.2f}", "RAW_FIGURE": raw_figure, "FEATURE_FIGURE": feature_figure,
        "FLOW_FIGURE": flow_figure, "DISTRIBUTION_FIGURE": distribution_figure,
        "WINDOW_AVG_FIGURE": window_avg_figure,
        "FEATURE_ROWS": ''.join(cells), "REGION_ROWS": regions_table, "RETENTION_TABLE": retention_table,
        "IDENTITY": provenance["pipeline_identity"][:12],
    }
    for key, value in substitutions.items():
        content = content.replace("{{" + key + "}}", value)
    if "{{" in content:
        raise ValueError("Unresolved placeholder in explainer HTML")
    destination = Path(destination) if destination is not None else ROOT / "docs" / "explainer.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8", newline="\n")
    print(f"Wrote {destination}")


TEMPLATE = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>From chords to an EEG table · Group 18</title>
<style>
:root{--ink:#25363d;--muted:#576c76;--teal:#1d7782;--paper:#f7f5ef;--line:#d9e1df}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1040px;margin:auto;padding:56px 30px 80px}h1,h2,h3{line-height:1.16;letter-spacing:-.025em}h1{font-size:clamp(38px,6vw,65px);max-width:850px;margin:18px 0}h2{font-size:31px;margin:0 0 20px}h3{font-size:21px}p{max-width:820px}a{color:var(--teal)}.eyebrow{font:700 12px/1.5 system-ui;text-transform:uppercase;letter-spacing:.15em;color:var(--teal)}.lead{font-size:21px;color:var(--muted);max-width:760px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:32px 0}.stat{border-top:3px solid var(--teal);padding:12px 0}.stat strong{display:block;font-size:31px;line-height:1.2}.stat span{font-size:13px;color:var(--muted)}.brief{background:#fff;border:1px solid var(--line);border-left:4px solid var(--teal);border-radius:8px;padding:16px 20px;margin:24px 0;font-size:15.5px}nav{display:flex;flex-wrap:wrap;gap:10px 22px;padding:18px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);font-size:14px}section{margin-top:64px;scroll-margin-top:20px}.num{display:block;font-size:12px;color:var(--teal);letter-spacing:.15em;margin-bottom:10px;font-weight:700}figure{overflow-x:auto;margin:26px 0;background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px}figure img{display:block;width:100%;height:auto}figure img,figure svg{min-width:640px}figcaption{font-size:13px;color:var(--muted);padding:12px 5px 0;line-height:1.5}svg{width:100%;height:auto;display:block}.note{border-left:4px solid #b66e29;background:#f0e8d9;padding:16px 20px;margin:22px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{background:#fff;border:1px solid var(--line);padding:22px;border-radius:10px}.card h3{margin-top:0}.decoder{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}.token{background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px 14px}.token code{display:inline-block;font-weight:700;margin-bottom:4px}.token span{display:block;font-size:13px;color:var(--muted);line-height:1.4}.steps{counter-reset:steps;padding:0;list-style:none}.steps li{counter-increment:steps;border-top:1px solid var(--line);padding:16px 0 16px 42px;position:relative}.steps li:before{content:counter(steps);position:absolute;left:0;top:17px;font-size:14px;font-weight:700;color:var(--teal)}details{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px 18px;margin:14px 0}summary{cursor:pointer;font-weight:650}details p{font-size:15px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.45;margin:15px 0}td,th{text-align:left;padding:10px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-size:12px}code{background:#e9efed;border-radius:4px;padding:2px 5px;font-size:.86em;overflow-wrap:anywhere}.formula{font-size:22px;text-align:center;padding:20px;background:white;border-radius:8px}footer{border-top:1px solid var(--line);margin-top:60px;padding-top:25px;font-size:13px;color:var(--muted)}@media(max-width:650px){main{padding:30px 18px}section{margin-top:44px}h2{font-size:27px}.grid,.decoder{grid-template-columns:1fr}.stats{gap:12px}.stat strong{font-size:25px}figure{padding:8px}body{font-size:16px}.formula{font-size:18px}}
</style><main>
<header><div class="eyebrow">CS1090A · Group 18 · A visual guide</div><h1>From chords<br>to an EEG table.</h1><p class="lead">A musical event becomes a short voltage trace. A few understandable averages turn that trace into a row we can analyze.</p>
<div class="stats"><div class="stat"><strong>{{PEOPLE}}</strong><span>participants retained</span></div><div class="stat"><strong>{{EVENTS}}</strong><span>retained events</span></div><div class="stat"><strong>21</strong><span>fixed EEG predictors</span></div></div>
<div class="brief"><b>In short:</b> Forty musicians listened to three-chord progressions while 64 scalp electrodes recorded EEG. Label-blind source, block-edge, electrode, ICA, and amplitude checks retain {{EVENTS}} middle-chord events from {{PEOPLE}} participants, each summarized by 21 baseline-corrected regional voltage averages (84% standards, 16% deviants). Milestone 3 will test whether those 21 numbers separate standard from deviant chords in held-out participants.</div>
<nav aria-label="Contents"><a href="#experiment">The experiment</a><a href="#cleaning">The cleaning</a><a href="#features">The averages</a><a href="#table">The table</a><a href="#evaluation">The next question</a><a href="#questions">Questions to expect</a></nav></header>
<section id="experiment"><span class="num">01 / THE EXPERIMENT</span><h2>The middle chord is the event.</h2><p>Participants heard three-chord sequences. Most middle chords were <b>standards</b>; some changed harmonic function or chord inversion. The target combines those two deviant categories. Electrodes recorded voltage at 2,048 samples per second.</p>
<figure><svg viewBox="0 0 900 260" role="img" aria-label="Schematic trial timeline, baseline and three early feature windows"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3z" fill="#6a7d85"/></marker></defs><line x1="58" y1="127" x2="850" y2="127" stroke="#6a7d85" stroke-width="2" marker-end="url(#arrow)"/><rect x="65" y="42" width="245" height="60" rx="8" fill="#e3e9e8"/><rect x="320" y="42" width="245" height="60" rx="8" fill="#1d7782"/><rect x="575" y="42" width="245" height="60" rx="8" fill="#e3e9e8"/><g text-anchor="middle" font-family="system-ui" font-size="18"><text x="187" y="79">First chord</text><text x="442" y="79" fill="white">Standard or deviant</text><text x="697" y="79">Third chord</text><text x="65" y="154">−400 ms</text><text x="320" y="154">0</text><text x="575" y="154">400 ms</text><text x="825" y="154">800 ms</text></g><rect x="256" y="180" width="64" height="25" fill="#b9c3c8"/><rect x="371" y="180" width="45" height="25" fill="#b66e29"/><rect x="416" y="180" width="64" height="25" fill="#80629e"/><rect x="480" y="180" width="95" height="25" fill="#4d885e"/><g font-family="system-ui" font-size="14" text-anchor="middle"><text x="282" y="228">Baseline</text><text x="474" y="228">80–150 · 150–250 · 250–400 ms</text></g></svg><figcaption>Schematic timing, not measured waveforms. Windows are nominal; sample alignment is accurate to about 1.95 ms, and zero-phase filters mix nearby activity.</figcaption></figure>
<p>The question: can 21 early EEG voltage features tell standard from deviant middle chords in participants the model was not trained on? It is <b>not</b> whether we have discovered a pure neural “surprise detector.” The conditions also differ acoustically and involve attention, decisions and button responses.</p></section>
<section id="cleaning"><span class="num">02 / THE CLEANING</span><h2>Keep the signal. Account for the losses.</h2><p>The pinned release contains {{GB}} GB. Its audit finds <b>{{SOURCE_COUNTS}}</b> recordings. A published checksum authenticates a file; it does not make a truncated file complete.</p>
<ol class="steps"><li><b>Verify and locate.</b> Check source bytes, units, channels and event timing. Keep only available portions of the two usable partial recordings.</li><li><b>Process within blocks.</b> Reduce the sampling rate to 256 Hz and filter. Never filter across a recording splice. The derived edge guard is {{GUARD}} seconds around each block boundary, plus the event window itself.</li><li><b>Repair persistent bad electrodes.</b> Detect dead channels and recurring noise, interpolate from neighboring electrodes, then average-reference.</li><li><b>Reduce artifacts with ICA.</b> Separate patterns on a dedicated fitting copy, classify them, and subtract high-confidence artifact components. The classification is imperfect and needs inspection.</li><li><b>Check all channels.</b> Make one intermittent-electrode repair pass. Reject epochs with extreme amplitude, flat channels or nonfinite samples. Apply the fixed participant-level retention and channel-count rules.</li></ol>
{{RAW_FIGURE}}{{FLOW_FIGURE}}
<details><summary>What do the thresholds mean?</summary><p>Epoch amplitude must stay between 1 and 150 µV peak-to-peak on every channel. At most six channels may be interpolated, and at least 60% of source- and edge-eligible events must pass signal QC. These are transparent operational rules, not universal biological constants. Passing a fixed 150 µV ceiling also does not guarantee an epoch is artifact-free: slower drift or lateral noise below 150 µV can still produce extreme window means. Thresholds are not adjusted to maximize cohort size or classification performance.</p></details>
<details><summary>Why is ICA more complicated than the later classifier?</summary><p>EEG is a mixture of scalp signals and artifacts. The cleaning step needs signal-processing tools before familiar tabular methods become appropriate. ICA uses a 1–100 Hz copy, common-average reference, seeded extended Picard and up to 30 components. Artifact labels above 0.8 confidence trigger removal. The analysis uses a separate 0.1–40 Hz band. Removing an artifact component can also remove some neural signal; the remaining data is not guaranteed artifact-free.</p></details>
<details><summary>Why run the bad-channel tests separately within blocks?</summary><p>The diagnostic filters must not cross discontinuities either. PyPREP tests each eligible block independently; a channel is considered persistently bad when flagged across at least half the tested duration. The low-amplitude dead-channel check is calculated across the available recording. Per-block flags and their weights are saved in the participant audit.</p></details></section>
<section id="features"><span class="num">03 / THE AVERAGES</span><h2>One waveform becomes three numbers.</h2><p>For each channel, subtract its mean from −100 to 0 ms. This makes later amplitudes relative to the immediately preceding activity. Then average across the electrodes in a region and across the samples in each time window.</p><div class="formula">Feature = average regional voltage in a window − regional baseline</div>
<div class="decoder" aria-label="How to read a feature column name like erp_FCmid_150_250_uv"><div class="token"><code>erp</code><span>Event-aligned voltage summary</span></div><div class="token"><code>FCmid</code><span>Scalp electrode region (frontocentral)</span></div><div class="token"><code>150_250</code><span>Window in ms after middle chord</span></div><div class="token"><code>uv</code><span>Microvolts relative to −100–0 ms baseline</span></div></div>
{{FEATURE_FIGURE}}
<figure><svg viewBox="0 0 800 380" role="img" aria-label="Schematic top view of seven scalp electrode regions"><ellipse cx="250" cy="195" rx="145" ry="155" fill="#f2f5f3" stroke="#809298" stroke-width="2"/><path d="M232 41 L250 16 L268 41 M104 165 Q70 190 104 220 M396 165 Q430 190 396 220" fill="none" stroke="#809298" stroke-width="2"/><g font-family="system-ui" font-size="15" text-anchor="middle"><text x="250" y="371" fill="#576c76">Back of head</text><text x="250" y="72" fill="#576c76">Front</text></g><g font-family="system-ui" font-size="15"><circle cx="175" cy="120" r="32" fill="#1d7782" opacity=".75"/><circle cx="325" cy="120" r="32" fill="#b66e29" opacity=".75"/><circle cx="250" cy="140" r="28" fill="#80629e" opacity=".75"/><circle cx="250" cy="210" r="28" fill="#4d885e" opacity=".75"/><circle cx="250" cy="285" r="28" fill="#c0615e" opacity=".75"/><circle cx="135" cy="225" r="28" fill="#5c7baf" opacity=".75"/><circle cx="365" cy="225" r="28" fill="#798743" opacity=".75"/><circle cx="470" cy="65" r="7" fill="#1d7782"/><text x="490" y="70">Lfront · left frontal</text><circle cx="470" cy="105" r="7" fill="#b66e29"/><text x="490" y="110">Rfront · right frontal</text><circle cx="470" cy="145" r="7" fill="#80629e"/><text x="490" y="150">FCmid · frontocentral</text><circle cx="470" cy="185" r="7" fill="#4d885e"/><text x="490" y="190">Cmid · central</text><circle cx="470" cy="225" r="7" fill="#c0615e"/><text x="490" y="230">Pmid · parietal</text><circle cx="470" cy="265" r="7" fill="#5c7baf"/><text x="490" y="270">Ltemp · left temporal</text><circle cx="470" cy="305" r="7" fill="#798743"/><text x="490" y="310">Rtemp · right temporal</text></g></svg><figcaption>Schematic electrode-group locations, not anatomical source estimates or measured scalp maps. Exact channel lists are below.</figcaption></figure>
<div class="grid"><div class="card"><h3>The worked example</h3><p>These three values are checked against the actual table row during generation of this guide.</p><table><thead><tr><th>FCmid window</th><th>Mean voltage</th></tr></thead><tbody>{{FEATURE_ROWS}}</tbody></table></div><div class="card"><h3>Seven regions × three windows</h3><p>The same calculation gives 21 predictors per event. Regional averaging makes the features easier to describe and reduces the number of inputs. It does not isolate a single brain area or eliminate noise.</p><p>No frequency-band powers, late windows or selected “best” features are included in this version.</p></div></div>
<details><summary>Which electrodes are in each region?</summary><table><thead><tr><th>Region</th><th>Electrodes</th></tr></thead><tbody>{{REGION_ROWS}}</tbody></table><p>Left/right frontal and temporal regions capture lateral coverage; FCmid, Cmid and Pmid cover frontocentral, central and parietal scalp regions. These are fixed electrode groups, not estimated anatomical sources.</p></details>
<div class="note"><b>“Early” does not mean “only surprise.”</b> The baseline occurs during the first chord. Filtering is noncausal, sample alignment is finite, and some responses occur before 400 ms. The features support offline condition classification, with these limitations.</div></section>
<section id="table"><span class="num">04 / THE TABLE</span><h2>Keep outcomes and metadata out of the predictors.</h2><div class="scroll"><table><thead><tr><th>Row identity</th><th>Outcome</th><th>Predictors</th><th>Other metadata</th></tr></thead><tbody><tr><td>Participant + event</td><td>0 standard / 1 deviant</td><td>21 <code>erp_*_uv</code> values</td><td>Block, trial, onset, subtype, musician group</td></tr></tbody></table></div><p>Only the explicit list of EEG features enters a future model. Trial order repeats across participants, so it can reveal the answer even when the participants themselves are separated. Stimulus names, responses and timing must not become predictors.</p>
{{DISTRIBUTION_FIGURE}}{{WINDOW_AVG_FIGURE}}<p>Deviants are {{DEVIANT}}% of the retained table. Always predicting “standard” can therefore look accurate. Across participants, condition window averages differ by roughly a microvolt while individual trials spread across several microvolts, so the next analysis should use ROC AUC and balanced accuracy alongside class counts.</p><details><summary>Did exclusions affect the classes equally?</summary><div class="scroll">{{RETENTION_TABLE}}</div><p>These are descriptive retention rates, not a test proving absence of selection bias. Exclusions may change which participants and conditions the table represents.</p></details></section>
<section id="evaluation"><span class="num">05 / THE NEXT QUESTION</span><h2>A new participant is the real test.</h2><p>The next milestone can start with regularized logistic regression and a simple decision tree. Each validation fold must hold out entire participants. Fit scaling, class weights or decision thresholds, tuning and other learned modeling transformations using only the training portion.</p>
<figure><svg viewBox="0 0 900 200" role="img" aria-label="Schematic grouped validation: all events of a participant stay together"><g font-family="system-ui" text-anchor="middle"><text x="270" y="30" font-size="18" fill="#1d7782">Training participants</text><text x="705" y="30" font-size="18" fill="#b66e29">Held-out participants</text><rect x="65" y="55" width="410" height="90" rx="12" fill="#e0eeeb"/><rect x="570" y="55" width="270" height="90" rx="12" fill="#f3e7d4"/><g font-size="18"><text x="150" y="108">A</text><text x="270" y="108">B</text><text x="390" y="108">C</text><text x="650" y="108">D</text><text x="760" y="108">E</text></g><text x="450" y="180" font-size="15" fill="#576c76">All events from one person stay on the same side.</text></g></svg><figcaption>Schematic example; letters are illustrative. No classifier is fitted in this MS2 foundation.</figcaption></figure>
<p>Cleaning uses each participant's complete unlabeled recording separately. That is an offline preprocessing assumption, not a real-time decoder. Previous exploration also means that a later split cannot retrospectively become a completely untouched test set.</p></section>
<section id="questions"><span class="num">06 / QUESTIONS TO EXPECT</span><h2>Explain the decision, then its limit.</h2>
<details><summary>What exactly does one row represent?</summary><p>One retained middle-chord event from one participant. Its target is the experimentally assigned standard/deviant category; its predictors are 21 fixed regional mean voltages.</p></details>
<details><summary>Why do fewer than 40 participants remain?</summary><p>Some source recordings are severely truncated. Further participants can fail fixed electrode or epoch-retention rules. The participant audit distinguishes those reasons and the event ledger accounts for every candidate.</p></details>
<details><summary>Why not keep every available feature?</summary><p>The early mean-voltage features provide a smaller, interpretable starting point. This is a design choice made before fitting this version's classifiers, not a search for the features with the best score.</p></details>
<details><summary>What does the preprocessing prove?</summary><p>It shows that verified source samples can be transformed consistently into an auditable table under the stated quality rules. It does not prove perfect artifact removal, model performance, a causal explanation, or generalization to other music and populations.</p></details>
<details><summary>What must happen before we can claim prediction works?</summary><p>Fit and evaluate models using participant-grouped validation, sensible imbalance-aware metrics and training-fold-only transformations. Compare against appropriate baselines and report variability and limitations.</p></details></section>
<footer><p><b>Read or run:</b> <a href="../README.md">Setup and quick start</a> · <a href="../ms2_data.ipynb">Data notebook</a> · <a href="methods.md">Detailed methods</a>. This HTML contains its figures and can be viewed offline. Wide figures scroll horizontally on small screens.</p><p><b>Sources &amp; software:</b> <a href="https://doi.org/10.82901/nemar.on003570">NEMAR on003570 v1.0.0</a> (OpenNeuro <a href="https://doi.org/10.18112/openneuro.ds003570.v1.0.0">ds003570</a>); <a href="https://doi.org/10.1016/j.neuroimage.2021.118458">Faller et al. (2021), <i>NeuroImage</i> 242, 118458</a>; <a href="https://doi.org/10.1177/0305735618779444">Goldman, Jackson &amp; Sajda (2020)</a>; <a href="https://doi.org/10.3389/fnins.2013.00267">MNE-Python (Gramfort et al., 2013)</a>; <a href="https://doi.org/10.3389/fninf.2015.00016">PREP / PyPREP (Bigdely-Shamlo et al., 2015)</a>; <a href="https://doi.org/10.1109/TSP.2018.2844203">Picard (Ablin et al., 2018)</a>; <a href="https://doi.org/10.1016/j.neuroimage.2019.05.026">ICLabel (Pion-Tonachini et al., 2019</a>; <a href="https://doi.org/10.21105/joss.04484">Li et al., 2022)</a>.</p><p><b>Generative AI disclosure:</b> Pipeline, tests, notebook, methods, walkthrough, earlier prototype handoff, and review with OpenAI Codex (GPT-6 Astra, GPT-6.1 Sol, GPT-6 Luna); earlier reference prototype, review, and notebook/README edits with Anthropic Claude Code (Claude Opus 5.5); cross-review, packaging, diagnostics, and citation updates with Google Antigravity (Gemini 4 Argon).</p><p>Measured figures and counts were generated from verified outputs. Pipeline identity: <code>{{IDENTITY}}</code>. Regenerate with <code>uv run python -m eeg_pipeline.explain</code>.</p></footer></main></html>'''

if __name__ == "__main__":
    build_explainer()
