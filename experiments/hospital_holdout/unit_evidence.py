"""Seal data-owner-confirmed ELAN export units; never a magnitude heuristic.

The explicit flag requires prior human confirmation of this export rule.
Paired EAFs, when available, must independently reproduce the parsed intervals.
The SHA-keyed output is private data and must not be committed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from data_tools import build_dataset as parser
from experiments.hospital_holdout.prepare import OLD_SHA, rows, sha, resolve, encoded


def eaf_check(path,events):
    root=ET.parse(path).getroot()
    header=root.find("HEADER")
    if header is None or header.get("TIME_UNITS") != "milliseconds":
        raise ValueError("paired EAF does not explicitly declare milliseconds")
    times={s.get("TIME_SLOT_ID"):float(s.get("TIME_VALUE"))/1000
           for s in root.findall("./TIME_ORDER/TIME_SLOT") if s.get("TIME_VALUE") is not None}
    tier=next((t for t in root.findall("TIER") if t.get("TIER_ID")=="default"),None)
    if tier is None:raise ValueError("paired EAF has no matching default tier")
    intervals=[]
    for a in tier.findall("./ANNOTATION/ALIGNABLE_ANNOTATION"):
        intervals.append((times[a.get("TIME_SLOT_REF1")],times[a.get("TIME_SLOT_REF2")],
                          parser.clean_label(a.findtext("ANNOTATION_VALUE"))[0]))
    measured=[(e["start_sec"],e["end_sec"],e["normalized_label"]) for e in events]
    intervals.sort(key=lambda r:r[:2]);measured.sort(key=lambda r:r[:2])
    if len(intervals)!=len(measured) or any(a[2]!=b[2] or
           max(abs(a[0]-b[0]),abs(a[1]-b[1]))>.001001 for a,b in zip(intervals,measured)):
        raise ValueError("TXT timestamps/labels do not replay paired EAF")


def generate(manifest_dir,records_path,data_root,confirmed):
    if not confirmed:raise ValueError("no owner-confirmed export rule; do not infer units")
    used=set()
    for split in OLD_SHA:
        f=Path(manifest_dir)/f"{split}.jsonl"
        if sha(f)!=OLD_SHA[split]:raise ValueError("source manifest SHA changed")
        used.update(r["record_id"] for r in rows(f))
    evidence,counts={},Counter()
    for record in rows(records_path):
        if record["record_id"] not in used or record["input_format"] != "elan_wide":continue
        path=resolve(record["source_txt_path"],data_root)
        if sha(path)!=record["source_txt_sha256"]:raise ValueError("original annotation SHA changed")
        text=path.read_text(encoding=record["text_encoding"])
        tokens=[token for line in text.splitlines() if line.startswith("TC")
                for match in parser.RANGE_RE.finditer(line) for token in match.groups() if ":" not in token]
        if not tokens:counts["clock_only"]+=1;continue
        # This applies the explicitly confirmed export convention, not a guess
        # from a threshold, duration, max value or prior model performance.
        unit="seconds" if any("." in token for token in tokens) else "milliseconds"
        events=parser.parse_events(text,"elan_wide",unit)
        candidates=list(path.parent.glob(path.stem+".eaf"))
        if path.parent.parent.is_dir():
            candidates.extend(q for q in path.parent.parent.rglob("*.eaf") if q.stem==path.stem)
        candidates=sorted(set(candidates))
        paired=[]
        for eaf in candidates:
            eaf_check(eaf,events)
            paired.append(sha(eaf))
        key=record["source_txt_sha256"]
        value={"unit":unit,"evidence_type":"data_owner_confirmation",
               "evidence":"User confirmed on 2026-10-10: decimal-second exports use seconds; integer-millisecond exports use milliseconds. No numeric-magnitude threshold used.",
               "paired_eaf_sha256":paired,"paired_eaf_replayed":bool(paired)}
        if key in evidence and evidence[key]!=value:raise ValueError("duplicate annotation SHA with contradictory evidence")
        evidence[key]=value
        counts[unit]+=1;counts["paired_eaf_replayed"]+=bool(paired)
    return evidence,dict(counts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest-dir",type=Path,required=True)
    p.add_argument("--records",type=Path,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--confirmed-decimal-seconds-integer-ms",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--write",action="store_true")
    a=p.parse_args()
    evidence,counts=generate(a.source_manifest_dir,a.records,a.data_root,a.confirmed_decimal_seconds_integer_ms)
    if a.write:
        repo=Path(__file__).resolve().parents[2];output=a.output.resolve()
        if repo==output or repo in output.parents:raise ValueError("private unit evidence cannot be stored in code checkout")
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("xb") as f:f.write(encoded(evidence))
    print(json.dumps({"annotation_evidence_count":len(evidence),"units":counts,"files_created":a.write}))


if __name__=="__main__":main()
