"""Named skeleton templates: a fixed set of landmarks per study segment, with
bones to draw and defaults for which landmarks are derived from the segment's
silhouette instead of tracked by appearance.

Derived specs (see tracker.py / silhouette.py):
    tip            tail tip: the end of the body midline
    midline:<f>    the point at fraction f (0 = head, 1 = tail tip) of the midline
    centroid       silhouette centroid
    ext:L / ext:R  the farthest-protruding extremity on each side (wing / patagium tips)
    ext:FL ext:FR ext:HL ext:HR   fore/hind extremities per side (feet)

Users can add templates as JSON files in <repo>/skeletons/*.json with the same
structure as the built-ins below.
"""
from __future__ import annotations

import json
from pathlib import Path

SKELETON_DIR = Path(__file__).resolve().parent.parent / "skeletons"

DERIVED_CHOICES: list[tuple[str, str]] = [
    ("tip", "tail tip (end of the midline)"),
    ("midline:0.25", "midline at 25 % of body length"),
    ("midline:0.5", "midline at 50 %"),
    ("midline:0.75", "midline at 75 %"),
    ("centroid", "silhouette centroid"),
    ("ext:L", "left extremity (wing / patagium tip)"),
    ("ext:R", "right extremity (wing / patagium tip)"),
    ("ext:FL", "fore-left extremity (foot)"),
    ("ext:FR", "fore-right extremity (foot)"),
    ("ext:HL", "hind-left extremity (foot)"),
    ("ext:HR", "hind-right extremity (foot)"),
]


def _t(name: str, head: str, landmarks: list[str], bones: list[tuple[str, str]],
       derived: dict[str, str], note: str = "") -> dict:
    return {"name": name, "head": head, "landmarks": list(landmarks),
            "bones": [list(b) for b in bones], "derived": dict(derived), "note": note}


BUILTIN: list[dict] = [
    _t("Lizard / iguana", "snout",
       ["snout", "eye", "neck", "shoulder_L", "shoulder_R", "hip_L", "hip_R", "tail_base",
        "tail_mid", "tail_tip", "foot_FL", "foot_FR", "foot_HL", "foot_HR"],
       [("snout", "eye"), ("eye", "neck"), ("neck", "shoulder_L"), ("neck", "shoulder_R"),
        ("shoulder_L", "hip_L"), ("shoulder_R", "hip_R"), ("hip_L", "tail_base"),
        ("hip_R", "tail_base"), ("tail_base", "tail_mid"), ("tail_mid", "tail_tip"),
        ("shoulder_L", "foot_FL"), ("shoulder_R", "foot_FR"), ("hip_L", "foot_HL"),
        ("hip_R", "foot_HR")],
       {"tail_mid": "midline:0.75", "tail_tip": "tip", "foot_FL": "ext:FL",
        "foot_FR": "ext:FR", "foot_HL": "ext:HL", "foot_HR": "ext:HR"},
       "Track the snout first: it anchors the midline that the tail and feet are derived from."),
    _t("Quadruped (generic)", "nose",
       ["nose", "head", "withers", "hip", "tail_base", "tail_tip", "paw_FL", "paw_FR",
        "paw_HL", "paw_HR"],
       [("nose", "head"), ("head", "withers"), ("withers", "hip"), ("hip", "tail_base"),
        ("tail_base", "tail_tip"), ("withers", "paw_FL"), ("withers", "paw_FR"),
        ("hip", "paw_HL"), ("hip", "paw_HR")],
       {"tail_tip": "tip", "paw_FL": "ext:FL", "paw_FR": "ext:FR", "paw_HL": "ext:HL",
        "paw_HR": "ext:HR"}),
    _t("Gliding mammal (flying squirrel)", "nose",
       ["nose", "eye_L", "eye_R", "wrist_L", "wrist_R", "ankle_L", "ankle_R",
        "patagium_L", "patagium_R", "tail_base", "tail_tip", "body_center"],
       [("nose", "eye_L"), ("nose", "eye_R"), ("eye_L", "wrist_L"), ("eye_R", "wrist_R"),
        ("wrist_L", "ankle_L"), ("wrist_R", "ankle_R"), ("ankle_L", "tail_base"),
        ("ankle_R", "tail_base"), ("tail_base", "tail_tip"), ("wrist_L", "patagium_L"),
        ("wrist_R", "patagium_R")],
       {"patagium_L": "ext:L", "patagium_R": "ext:R", "tail_tip": "tip",
        "body_center": "centroid"},
       "Patagium tips are the widest points of the silhouette on each side."),
    _t("Flying lizard (Draco)", "snout",
       ["snout", "neck", "wing_tip_L", "wing_tip_R", "hip", "tail_base", "tail_mid", "tail_tip",
        "foot_FL", "foot_FR", "foot_HL", "foot_HR"],
       [("snout", "neck"), ("neck", "hip"), ("hip", "tail_base"), ("tail_base", "tail_mid"),
        ("tail_mid", "tail_tip"), ("neck", "wing_tip_L"), ("neck", "wing_tip_R"),
        ("neck", "foot_FL"), ("neck", "foot_FR"), ("hip", "foot_HL"), ("hip", "foot_HR")],
       {"wing_tip_L": "ext:L", "wing_tip_R": "ext:R", "tail_mid": "midline:0.75",
        "tail_tip": "tip"}),
    _t("Undulating body (fish / eel / swimming)", "snout",
       ["snout", "body_25", "body_50", "body_75", "tail_tip"],
       [("snout", "body_25"), ("body_25", "body_50"), ("body_50", "body_75"),
        ("body_75", "tail_tip")],
       {"body_25": "midline:0.25", "body_50": "midline:0.5", "body_75": "midline:0.75",
        "tail_tip": "tip"},
       "Only the snout is tracked by appearance; the rest is the silhouette midline."),
    _t("Bird / bat (wings)", "beak",
       ["beak", "head", "wing_tip_L", "wing_tip_R", "wrist_L", "wrist_R", "tail_tip",
        "body_center"],
       [("beak", "head"), ("head", "body_center"), ("body_center", "wrist_L"),
        ("body_center", "wrist_R"), ("wrist_L", "wing_tip_L"), ("wrist_R", "wing_tip_R"),
        ("body_center", "tail_tip")],
       {"wing_tip_L": "ext:L", "wing_tip_R": "ext:R", "tail_tip": "tip",
        "body_center": "centroid"}),
]


EXT_ROLES = ("L", "R", "FL", "FR", "HL", "HR")
_SPEC_HELP = "use tip, centroid, midline:<0 to 1> or ext:L / ext:R / ext:FL / ext:FR / ext:HL / ext:HR"


def validate_spec(spec) -> tuple[bool, str]:
    """(ok, what is wrong in plain words) for a derived-landmark spec (I62).
    Nothing downstream checks a spec: 'midline:50' (read off the '50 %' label)
    was clipped to the tail tip and exported as a full-confidence copy of it,
    and a typo ('ext:LF') gave a column that is always empty, unexplained."""
    s = str(spec).strip()
    if s in ("tip", "centroid"):
        return True, ""
    if s.startswith("ext:"):
        if s[4:] in EXT_ROLES:
            return True, ""
        return False, f"'{s}' is not an extremity: {_SPEC_HELP}"
    if s.startswith("midline:"):
        try:
            f = float(s.split(":", 1)[1])
        except ValueError:
            f = float("nan")
        if 0.0 <= f <= 1.0:
            return True, ""
        if 1.0 < f <= 100.0:
            return False, (f"'{s}': the midline position is a fraction from 0 (head) to 1 (tail tip), "
                           f"not a percentage - did you mean midline:{f / 100:g}?")
        return False, f"'{s}': the midline position must be a number from 0 (head) to 1 (tail tip), e.g. midline:0.5"
    return False, f"'{s}' is not a silhouette rule: {_SPEC_HELP}"


def validate_template(t: dict) -> tuple[dict, list[str]]:
    """A cleaned copy of a skeleton template plus the problems found, in plain
    words (I62). A bad derived spec is dropped (that landmark is then tracked by
    appearance instead of silently exporting a wrong or empty column); a bone
    that is not a pair of known landmarks is dropped (session.bones() raised on
    it); a head that is not a landmark falls back to the first landmark."""
    problems: list[str] = []
    marks = []
    for m in (t.get("landmarks") or []):
        m = str(m).strip()
        if not m:
            continue
        if m in marks:       # (G122) a name twice would become a stray "head (2)" point
            problems.append(f"landmark '{m}' is listed more than once; the repeat is ignored")
            continue
        marks.append(m)
    out = dict(t)
    out.update({"name": str(t.get("name") or "custom"), "landmarks": marks,
                "note": str(t.get("note") or ""), "derived": {}, "bones": []})
    head = str(t.get("head") or "").strip()
    if marks and head not in marks:
        if head:
            problems.append(f"head landmark '{head}' is not one of the landmarks; '{marks[0]}' is used")
        head = marks[0]
    out["head"] = head
    derived = t.get("derived") or {}
    if not isinstance(derived, dict):
        problems.append("'derived' must map landmark names to silhouette rules; ignored")
        derived = {}
    for nm, spec in derived.items():
        if nm not in marks:
            problems.append(f"'{nm}' has a silhouette rule but is not a landmark; ignored")
            continue
        if not str(spec).strip():
            continue
        ok, why = validate_spec(spec)
        if ok:
            out["derived"][nm] = str(spec).strip()
        else:
            problems.append(f"{nm}: {why} ('{nm}' will be tracked by appearance instead)")
    for b in t.get("bones") or []:
        if (isinstance(b, (list, tuple)) and len(b) == 2 and all(isinstance(x, str) for x in b)
                and b[0] in marks and b[1] in marks):
            out["bones"].append([b[0], b[1]])
        else:
            problems.append(f"bone {b!r} is not a pair of landmark names; ignored")
    return out, problems


def load_user_templates(problems: list[str] | None = None) -> list[dict]:
    """Templates from skeletons/*.json, validated. What is wrong with a file
    (unreadable JSON, a bad spec, a bad bone) is appended to `problems` as a
    sentence naming the file, instead of the file silently vanishing (I62)."""
    out = []
    if SKELETON_DIR.exists():
        for p in sorted(SKELETON_DIR.glob("*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                if problems is not None:
                    problems.append(f"skeletons/{p.name} could not be read ({e}); it is not in the menu")
                continue
            if not (isinstance(d, dict) and d.get("landmarks")):
                if problems is not None:
                    problems.append(f"skeletons/{p.name} has no 'landmarks' list; it is not in the menu")
                continue
            d.setdefault("name", p.stem)
            clean, found = validate_template(d)
            if problems is not None:
                problems.extend(f"skeletons/{p.name}: {msg}" for msg in found)
            out.append(clean)
    return out


def user_template_problems() -> list[str]:
    """What is wrong with the files in skeletons/, one sentence each (for the app to show)."""
    problems: list[str] = []
    load_user_templates(problems)
    return problems


def all_templates() -> list[dict]:
    return BUILTIN + load_user_templates()


def template_by_name(name: str) -> dict | None:
    return next((t for t in all_templates() if t["name"] == name), None)


def user_template_path(name: str) -> Path:
    """The file skeletons/<name>.json a template of that name is saved in."""
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip() or "skeleton"
    return SKELETON_DIR / f"{safe}.json"


def save_user_template(t: dict, overwrite: bool = False) -> Path:
    """Write `t` to skeletons/<name>.json. A template of that name already there
    is NOT replaced unless `overwrite` (G122): FileExistsError (an OSError) says
    so, and the caller asks the user before trying again with overwrite=True."""
    p = user_template_path(t["name"])
    SKELETON_DIR.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        raise FileExistsError(f"skeletons/{p.name} already exists")
    p.write_text(json.dumps(t, indent=2), encoding="utf-8")
    return p
