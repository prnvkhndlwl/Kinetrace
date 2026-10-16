"""Camera timing from the wand (I271 `wandsync.py`, I300-I304) on a synthetic rig with KNOWN fractional
offsets (`_synth_wand.py`: four 1280 x 720 cameras at 100 fps, a 0.5 m wand, 0.3 px noise; the project's
offsets wrong by +0.4 / -0.7 frame unless a section says otherwise).

Ported from the parked branch (I271):
  A. the robust estimate: wand ends swapped on 40 frames of one camera drag the plain estimate, not the
     robust one; on clean tracks the two are the same;
  B. the circularity: a calibration fitted at wrong offsets has absorbed part of the error, so ONE estimate
     with it fixed under-corrects; the refine loop recovers every offset and leaves the project alone;
  C. the loop with swapped ends;  D. a wand always moved the same way at the same speed: nothing reads good;
  E. a frame-rate drift: "drift", the halves disagree, the row is poor;  F. a dropped frame: "jump";
  G. the speed screen names the camera that is off;  H. two cameras;  I. cancel.
New:
  J. (I302) each camera is judged on its OWN cells: a camera in a few % of the cells reads sharp with
     enough of them, at most weak with fewer; the verdict never changes the search;
  K. (I300) the whole-frame check never turns a fraction of a frame into a whole-frame suggestion,
     whichever cameras are ticked, and names the fraction; with camera 1 unticked the median rule;
  L. (I301) a camera with plenty of shared wand frames but no clear peak gets the real reason;
  M. the other cameras' prediction of each camera's wand ends: a camera a frame off reads poor / ok and
     good after the refine; two cameras cannot predict;
  N. camera 1 unticked: the run's clock never moves; O. a middle camera unticked: every number lands on
     its own project camera;  P. a camera 2 frames off plus fractions on the others;
  Q. RunView: a run's cameras at other offsets, and nothing else;
  R. (I304) a mark clicked on one frame is found at a fractional offset; whole frames unchanged;
  S. everything the checks put in a report is strict JSON.

Run: .venv\\Scripts\\python.exe tests\\verify_wand_sync.py
"""
import itertools
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.stdout.reconfigure(errors="replace")

import numpy as np

import _synth_wand as sw
from kinetrace import wanddata as wd
from kinetrace.calib import Calibration, estimate_offsets

A, B = "wand A", "wand B"
REFINE_TOL = 0.05            # frames: the loop's offsets against the truth
failures = []


def check(name):
    def deco(fn):
        try:
            fn()
            print(f"  ok  {name}")
        except Exception as e:      # noqa: BLE001
            import traceback
            traceback.print_exc()
            failures.append(name)
            print(f"  FAIL {name}: {e}")
        return fn
    return deco


def ws():
    """The module under test, imported inside each section (the calib / wanddata sections run without it)."""
    from kinetrace import wandsync
    return wandsync


def err(offs, truth):
    return np.abs(np.asarray(offs) - np.asarray(truth["offsets"]))


def calibrator(proj, views=None):
    # the focal lengths as known starting values (still refined): ~0.3 s per calibration
    return sw.calibrate_fn(proj, focal=list(sw.FOCAL[:proj.n_views]), views=views)


def frames_of(proj):
    _, fr = wd.collect_wand(proj, A, B, max_frames=10 ** 9)
    return int(fr.min()), int(fr.max())


@check("A. robust estimate: wand ends swapped on 40 frames drag the plain estimate, not the robust one")
def _():
    proj, truth = sw.build(swaps=40)
    cal = Calibration(truth["cams"], "m", "truth")          # even with the TRUE cameras
    rng = frames_of(proj)
    plain, _, _ = estimate_offsets(proj.sessions, cal, proj.rates, proj.offsets, rng, names=[A, B])
    rep: dict = {}
    robust, _, _ = estimate_offsets(proj.sessions, cal, proj.rates, proj.offsets, rng, names=[A, B], report=rep,
                                    robust=True)
    print(f"      plain off by {np.round(err(plain, truth), 3)}, robust by {np.round(err(robust, truth), 3)} "
          f"({rep['n_set_aside']} outlying cells)")
    assert err(plain, truth).max() > 0.15, plain
    assert err(robust, truth).max() < REFINE_TOL, robust
    assert rep["n_set_aside"] >= 40 and rep["verdict"] == "reliable", rep
    # on clean tracks the robust estimate IS the plain one
    proj, truth = sw.build()
    a, _, _ = estimate_offsets(proj.sessions, cal, proj.rates, proj.offsets, rng, names=[A, B])
    rep = {}
    b, _, _ = estimate_offsets(proj.sessions, cal, proj.rates, proj.offsets, rng, names=[A, B], report=rep,
                               robust=True)
    assert a == b and rep["n_set_aside"] == 0, (a, b, rep["n_set_aside"])


@check("B. circularity: one estimate against a calibration fitted at wrong offsets under-corrects; the loop does not")
def _():
    W = ws()
    proj, truth = sw.build()
    rng = frames_of(proj)
    # the true cameras: one estimate is enough (what is left is the floor: linear interpolation between frames
    # of a curved path)
    good, _ = W.estimate(proj, Calibration(truth["cams"], "m", "truth"), A, B, proj.offsets, rng)
    e0 = err(good, truth)
    assert e0.max() < REFINE_TOL, e0
    # the calibration made at the wrong offsets: the same estimate stops short
    res = calibrator(proj)(proj.offsets)
    one, _ = W.estimate(proj, res.to_calibration(), A, B, proj.offsets, rng)
    e1 = err(one, truth)
    print(f"      injected 0.4 / 0.7: with the true cameras one estimate leaves {np.round(e0, 3)}, with the cameras "
          f"fitted at those offsets {np.round(e1, 3)}")
    assert e1.max() > 0.06 and e1.sum() > e0.sum() + 0.08, (e0, e1)
    for c in (1, 2):
        assert np.sign(one[c] - proj.offsets[c]) == np.sign(truth["offsets"][c] - proj.offsets[c]), "wrong way"
    start = list(proj.offsets)
    out = W.refine(proj, A, B, calibrator(proj))
    e2 = err(out["after"], truth)
    print(f"      refine: {len(out['rounds'])} rounds, moves {[round(r['moved'], 3) for r in out['rounds']]}, left "
          f"{np.round(e2, 3)}; wand score {out['wand_score_before']:.2f} -> {out['wand_score_after']:.2f} %")
    assert out["converged"], out["rounds"]
    assert e2.max() < REFINE_TOL, e2
    assert e2.sum() < 0.5 * e1.sum() and np.all(e2 <= e0 + 0.02), (e0, e1, e2)
    assert out["verdict"] == "good", (out["why"], {c: r["why"] for c, r in out["rows"].items()})
    assert out["wand_score_after"] < out["wand_score_before"]
    assert out["rows"][1]["tick"] and out["rows"][2]["tick"], out["rows"]
    for c, r in out["rows"].items():
        assert r["sharpness"] == "sharp" and r["joint"] == "sharp" and r["split"] == "agree", (c, r)
        assert r["reproj_after"] <= r["reproj_before"] + 1e-6, (c, r)
    assert out["after"][0] == 0.0 and proj.offsets == start, "the project must not change"


@check("C. refine with swapped ends: the robust estimate keeps the loop on the truth")
def _():
    W = ws()
    proj, truth = sw.build(swaps=40)
    out = W.refine(proj, A, B, calibrator(proj))
    e = err(out["after"], truth)
    print(f"      left {np.round(e, 3)}, verdict {out['verdict']}")
    assert e.max() < REFINE_TOL, e
    assert out["verdict"] == "good", out["why"]


@check("D. a wand always moved the same way at the same speed: no camera reads good")
def _():
    W = ws()
    proj, truth = sw.build(straight=True)
    out = W.refine(proj, A, B, calibrator(proj))
    words = {c: (r["verdict"], r["joint"]) for c, r in out["rows"].items()}
    print(f"      rows {words}, joint {({c: round(v, 3) for c, v in out['joint'].items()})}")
    assert out["verdict"] != "good", out["why"]
    for c, r in out["rows"].items():
        assert r["verdict"] != "good" and not r["tick"], (c, r)
        assert r["joint"] in ("flat", "weak"), (c, r)
    d = W.diagnose(proj, out["res_before"].to_calibration(), A, B)
    assert d["speed"]["verdict"] == "ok" and "nearly one speed" in d["speed"]["why"], d["speed"]


@check("E. a frame-rate drift: the window check says drift, the halves disagree, the row is poor")
def _():
    W = ws()
    proj, truth = sw.build(error=(0, 0, 0, 0), rate_true=[1, 1, 1.0008, 1])
    out = W.refine(proj, A, B, calibrator(proj))
    r = out["rows"][2]
    w = out["windows"]["per_camera"][2]
    print(f"      cam3 windows {np.round(w['windows'], 3)} -> {w['pattern']}; halves {r['split']}; {r['verdict']}")
    assert w["pattern"] == "drift", w
    assert 0.0004 < w["drift_per_frame"] < 0.0012, w["drift_per_frame"]
    assert r["split"] == "disagree" and r["verdict"] == "poor" and not r["tick"], r
    for c in (1, 3):
        assert out["windows"]["per_camera"][c]["pattern"] == "steady", out["windows"]["per_camera"][c]
    assert out["verdict"] == "poor"
    # the same drift is seen without refining (at the project's offsets)
    d = W.diagnose(proj, out["res_before"].to_calibration(), A, B)
    assert d["windows"]["per_camera"][2]["pattern"] == "drift", d["windows"]["per_camera"][2]


@check("F. a dropped frame: the window check says jump, by about one frame")
def _():
    W = ws()
    proj, truth = sw.build(error=(0, 0, 0, 0), dropped=(1, 600))
    res = calibrator(proj)(proj.offsets)
    d = W.diagnose(proj, res.to_calibration(), A, B)
    w = d["windows"]["per_camera"][1]
    print(f"      cam2 windows {np.round(w['windows'], 3)} -> {w['pattern']}")
    assert w["pattern"] == "jump" and -1.15 < w["jump"] < -0.75, w
    assert "dropped or repeated frame" in w["why"]


@check("G. the speed check: clean offsets read good, a 0.5-frame error names its camera")
def _():
    W = ws()
    proj, truth = sw.build(error=(0, 0, 0, 0))
    res = calibrator(proj)(proj.offsets)
    d = W.diagnose(proj, res.to_calibration(), A, B)
    print(f"      clean: ratio {d['speed']['ratio']:.2f}, windows {d['windows']['verdict']}, verdict {d['verdict']}")
    assert d["speed"]["verdict"] == "good" and d["windows"]["verdict"] == "good" and d["verdict"] == "good", d
    assert all(abs(v) < 0.05 for v in d["off_by"].values()), d["off_by"]
    assert d["speed"]["fastest_frame"] is not None
    proj, truth = sw.build(error=(0, 0, 0, -0.5))
    res = calibrator(proj)(proj.offsets)
    d = W.diagnose(proj, res.to_calibration(), A, B)
    sp = d["speed"]
    print(f"      cam4 0.5 off: ratio {sp['ratio']:.2f}, leave-one-out {({c: round(v, 2) for c, v in sp['loo_ratio'].items()})}")
    assert sp["verdict"] in ("ok", "poor") and sp["implicated"] == 3, sp
    assert "cam4 is the camera most implicated" in sp["why"], sp["why"]
    assert d["off_by"][3] > 0.2, d["off_by"]


@check("H. two cameras: the loop and the checks run; the speed check cannot name one")
def _():
    W = ws()
    proj, truth = sw.build(error=(0, 0.4), true_offsets=sw.TRUE_OFFSETS[:2])
    out = W.refine(proj, A, B, calibrator(proj))
    e = err(out["after"], {"offsets": sw.TRUE_OFFSETS[:2]})
    print(f"      left {np.round(e, 3)}, verdict {out['verdict']}")
    assert e.max() < REFINE_TOL, e
    d = W.diagnose(proj, out["res_before"].to_calibration(), A, B)
    assert d["speed"]["implicated"] is None
    if d["speed"]["verdict"] != "good":
        assert "two cameras" in d["speed"]["why"], d["speed"]["why"]


@check("I. cancel: the loop stops and says so")
def _():
    W = ws()
    proj, truth = sw.build()
    calls = []

    def calibrate(o, hint=None):
        calls.append(1)
        return calibrator(proj)(o, hint)
    out = W.refine(proj, A, B, calibrate, cancelled=lambda: len(calls) >= 2)
    assert out == {"cancelled": True}, out
    assert len(calls) == 2, calls


@check("J. (I302) a camera in a few % of the cells is judged on its OWN cells")
def _():
    # camera 4 films the wand only for 70 of the 1200 reference frames, with 1 px of picture noise
    proj, truth = sw.build(error=(0, 0, 0, 0), noise=1.0, window=(3, 300, 370))
    cal = Calibration(truth["cams"], "m", "truth")
    rng = frames_of(proj)
    rep: dict = {}
    offs, before, after = estimate_offsets(proj.sessions, cal, proj.rates, proj.offsets, rng, names=[A, B],
                                           report=rep)
    again = estimate_offsets(proj.sessions, cal, proj.rates, proj.offsets, rng, names=[A, B])
    assert (offs, before, after) == again, "the verdict must never change the search"
    own, sh = rep["own_cells"], rep["sharpness"]
    share = own[3] / max(rep["n_scored_cells"], 1)
    print(f"      cam4 in {100 * share:.1f} % of the scored cells ({own[3]}): sharpness {sh[3]:.3f} -> "
          f"{rep['per_view'][3]}; others {[rep['per_view'][c] for c in (1, 2)]}")
    assert share < 0.10, share
    assert own[3] >= 30 and rep["per_view"][3] == "sharp", (own[3], sh[3], rep["per_view"])
    # fewer than MIN_OWN_CELLS of its own: never better than weak
    proj, truth = sw.build(error=(0, 0, 0, 0), noise=1.0, window=(3, 300, 306))
    rep = {}
    estimate_offsets(proj.sessions, cal, proj.rates, proj.offsets, frames_of(proj), names=[A, B], report=rep)
    print(f"      7 frames: {rep['own_cells'][3]} own cells -> {rep['per_view'][3]}")
    assert rep["own_cells"][3] < 30 and rep["per_view"][3] in ("weak", "flat", "none"), rep["per_view"]


@check("K. (I300) the whole-frame check never rounds a fraction into a whole-frame suggestion")
def _():
    # 1) a 6-camera rig whose errors are fractions of a frame, like the measured 10-camera rig
    true6 = [0.0, 3.30, -2.60, 1.15, 0.70, -1.40]
    proj, truth = sw.build(error=(0, 0.6, 0.3, -0.35, 0.45, 0.0), true_offsets=true6)
    tc = wd.timing_check(proj, A, B, list(range(6)))
    print(f"      all six: {tc.verdict}; fitted {({c: round(v, 2) for c, v in tc.fitted.items() if v is not None})}; "
          f"fraction {({c: round(v, 2) for c, v in tc.fraction.items()})}")
    assert not tc.suggestions, tc.suggestions
    assert not any("no offset needs to change" in ln for ln in tc.lines), tc.lines
    n_sub = 0
    for k in range(3, 7):
        for views in itertools.combinations(range(6), k):
            f = wd._fit_shifts(tc.pairs, list(views), proj.rates)
            if not f.consistent:
                continue
            n_sub += 1
            for c, d in f.suggestions.items():
                assert abs(f.fitted[c]) >= wd.WHOLE_MIN, (views, c, f.fitted[c], d)
            for c, v in f.fitted.items():
                if v is not None and wd.FRACTION_MIN <= abs(v) < wd.WHOLE_MIN:
                    assert c in f.fraction and c not in f.suggestions, (views, c, v)
    print(f"      {n_sub} consistent subsets of 3-6 cameras: no whole-frame suggestion below {wd.WHOLE_MIN}")
    # 2) camera 1 unticked: the fits are made relative to the first timed camera; the MEDIAN of the unrounded
    #    fits is "none" (the most common ROUNDED shift made the reference read +1 here)
    sug, frac, shift = wd._classify_shifts({1: 0.0, 2: -0.6, 3: -0.55, 4: -0.51}, clock_ticked=False)
    assert not sug and 1 in frac, (sug, frac, shift)
    #    two cameras ticked, neither camera 1: a clear whole-frame pair moves ONE camera, never half each
    #    (review: the plain median split +1 into -0.5 / +0.5 and +3 into -1 / +2)
    for best in (1, -1, 3):
        pair = [{"cams": (1, 2), "rows": 200, "best": best, "used": True}]
        f = wd._fit_shifts(pair, [1, 2], [1.0] * 4)
        assert f.suggestions == {2: best} and not f.fraction, (best, f.fitted, f.suggestions, f.fraction)
    # 3) whole frames are still suggested as before
    sug, frac, _ = wd._classify_shifts({0: 0.0, 1: -3.02, 2: 0.08}, clock_ticked=True)
    assert sug == {1: -3} and not frac, (sug, frac)


@check("L. (I301) plenty of shared wand frames but no clear peak: the real reason")
def _():
    proj, truth = sw.build(error=(0, 0, 0, 0), noise=[0.3, 0.3, 0.3, 9.0])
    tc = wd.timing_check(proj, A, B, list(range(4)))
    rows4 = max(r["rows"] for r in tc.pairs if 3 in r["cams"])
    mine = [ln for ln in tc.lines if "cam4" in ln]
    print(f"      cam4: {rows4} shared wand instants at most; {mine}")
    assert rows4 >= 100 and tc.shift[3] is None, (rows4, tc.shift)
    assert any("cannot check the timing of cam4" in ln for ln in tc.lines), tc.lines
    assert not any("too few" in ln for ln in mine), mine
    assert any("clear" in ln for ln in mine), mine


@check("M. the other cameras' prediction of each camera's wand ends")
def _():
    W = ws()
    proj, truth = sw.build(error=(0, 0, 0, 1.0))
    res = calibrator(proj)(proj.offsets)
    pe = W.prediction_error(proj, res.to_calibration(), A, B)
    med = {c: pe["per_camera"][c]["median_px"] for c in range(4)}
    print(f"      cam4 a frame off: medians {({c: round(v, 2) for c, v in med.items()})} px, "
          f"{({c: r['verdict'] for c, r in pe['per_camera'].items()})}")
    assert med[3] > 3 * float(np.median([med[c] for c in range(3)])), med
    assert pe["per_camera"][3]["verdict"] in ("ok", "poor"), pe["per_camera"][3]
    out = W.refine(proj, A, B, calibrator(proj))
    pe2 = W.prediction_error(proj, out["res_after"].to_calibration(), A, B, offsets=out["after"])
    print(f"      after the refine: {({c: (round(r['median_px'], 2), r['verdict']) for c, r in pe2['per_camera'].items()})}")
    assert all(r["verdict"] == "good" for r in pe2["per_camera"].values()), pe2
    proj2, _ = sw.build(error=(0, 0.4), true_offsets=sw.TRUE_OFFSETS[:2])
    res2 = calibrator(proj2)(proj2.offsets)
    pe3 = W.prediction_error(proj2, res2.to_calibration(), A, B)
    assert all(r["n"] == 0 for r in pe3["per_camera"].values()) and "3 cameras" in pe3["why"], pe3


@check("N. camera 1 unticked: the run's clock never moves, the others are right relative to it")
def _():
    W = ws()
    proj, truth = sw.build(error=(0, 0.4, -0.7, 0.3))
    views = [1, 2, 3]
    out = W.refine(proj, A, B, calibrator(proj, views=views), views=views)
    t = truth["offsets"]
    rel = [(out["after"][c] - out["after"][1]) - (t[c] - t[1]) for c in (2, 3)]
    print(f"      clock cam2 kept at {out['after'][1]:+.3f}; cam3 / cam4 relative to it off by {np.round(rel, 3)}")
    assert out["after"][0] == proj.offsets[0] and out["after"][1] == proj.offsets[1], out["after"]
    assert max(abs(x) for x in rel) < REFINE_TOL, rel
    assert sorted(out["rows"]) == [2, 3], sorted(out["rows"])


@check("O. a middle camera unticked (views 0, 2, 3): every number lands on its own project camera")
def _():
    W = ws()
    proj, truth = sw.build(error=(0, 0.4, -0.7, 0.3))
    views = [0, 2, 3]
    out = W.refine(proj, A, B, calibrator(proj, views=views), views=views)
    e = [abs(out["after"][c] - truth["offsets"][c]) for c in (2, 3)]
    print(f"      cam3 / cam4 left {np.round(e, 3)}; cam2 (not in the run) kept at {out['after'][1]:+.3f}")
    assert out["after"][1] == proj.offsets[1], out["after"]
    assert max(e) < REFINE_TOL, e
    assert sorted(out["rows"]) == [2, 3] and out["rows"][2]["old"] == proj.offsets[2], out["rows"]
    cal = out["res_after"].to_calibration()
    d = W.diagnose(proj, cal, A, B, views=views, offsets=out["after"])
    assert set(d["off_by"]) <= {2, 3} and set(d["windows"]["per_camera"]) <= {2, 3}, d
    pe = W.prediction_error(proj, cal, A, B, views=views, offsets=out["after"])
    assert set(pe["per_camera"]) == {0, 2, 3}, pe["per_camera"]


@check("P. a camera 2 frames off plus fractions on the others: all recovered")
def _():
    W = ws()
    proj, truth = sw.build(error=(0, 2.0, -0.4, 0.6))
    out = W.refine(proj, A, B, calibrator(proj))
    e = err(out["after"], truth)
    print(f"      {len(out['rounds'])} rounds, left {np.round(e, 3)}")
    assert e.max() < REFINE_TOL, e


@check("Q. RunView: a run's cameras at other offsets, and nothing else")
def _():
    proj, truth = sw.build()
    offs = [0.0, proj.offsets[1] + 0.37, proj.offsets[2] - 0.21, proj.offsets[3] + 0.5]
    rv = wd.RunView(proj, [0, 2, 3], offs)
    assert rv.n_views == 3 and rv.views == [0, 2, 3] and rv.offsets == [offs[0], offs[2], offs[3]], rv.offsets
    assert rv.names == ["cam1", "cam3", "cam4"] and rv.name(1) == "cam3"
    uv1, fr1 = wd.collect_wand(rv, A, B, max_frames=10 ** 9)
    proj2, _ = sw.build()
    for c in range(1, 4):
        proj2.set_offset(c, offs[c])
    uv2, fr2 = wd.collect_wand(proj2, A, B, max_frames=10 ** 9, views=[0, 2, 3])
    assert np.array_equal(fr1, fr2) and np.array_equal(uv1, uv2, equal_nan=True)
    for attr in ("local_index", "reference_index", "map_frame"):
        try:
            getattr(rv, attr)
        except AttributeError:
            continue
        raise AssertionError(f"RunView.{attr} must not exist (it would use the project's own offsets)")
    rt = wd.retimed(proj, offs)
    assert rt.views == [0, 1, 2, 3] and rt.offsets == offs
    assert wd.retimed(proj, None).offsets == list(proj.offsets)


@check("R. (I304) a mark clicked on one frame is found at a fractional offset")
def _():
    proj, truth = sw.build(marks_at=400, error=(0, 0, 0, 0), true_offsets=[0.0, 3.0, -2.0, 1.0])  # whole frames
    assert all(o == round(o) for o in proj.offsets), proj.offsets
    t = 400
    a = wd.collect_at(proj, "origin", t)
    s = wd.collect_static_at(proj, "origin", t)
    assert np.isfinite(s).all() and np.array_equal(a, s), (a, s)        # whole frames: bit-identical
    o = list(proj.offsets)
    rv = wd.retimed(proj, [o[0], o[1] + 0.4, o[2] - 1.3, o[3] + 2.6])
    assert np.isnan(wd.collect_at(rv, "origin", t)[1]).all(), "interpolation needs two clicked frames"
    s2 = wd.collect_static_at(rv, "origin", t)
    assert np.array_equal(s2, s), (s2, s)                                # the same clicks
    far = wd.retimed(proj, [o[0], o[1] + 4.0, o[2], o[3]])
    assert np.isnan(wd.collect_static_at(far, "origin", t)[1]).all(), "more than 3 frames away: not this click"
    for nm in ("X", "Y"):
        assert np.isfinite(wd.collect_static_at(rv, nm, t)).all(), nm
    # a mark placed on BOTH neighbouring frames (tracked, held or clicked twice): interpolated exactly as
    # main's collect_at did, so Calibrate now at a fractional project offset is bit-identical (review)
    s1 = proj.sessions[1]
    jm = s1.pid_by_name("origin")
    fm = int(np.nonzero(s1.tracked[:, jm])[0][0])
    s1.tracks[fm + 1, jm] = s1.tracks[fm, jm] + (3.0, -2.0)
    s1.tracked[fm + 1, jm] = True
    s1.manual[fm + 1, jm] = True
    s1._touch()
    rv2 = wd.retimed(proj, [o[0], o[1] + 0.4, o[2], o[3]])
    a3 = wd.collect_at(rv2, "origin", t)
    assert np.isfinite(a3[1]).all(), a3
    assert np.array_equal(wd.collect_static_at(rv2, "origin", t), a3), (wd.collect_static_at(rv2, "origin", t), a3)


@check("S. what the checks put in a report is strict JSON")
def _():
    W = ws()
    proj, truth = sw.build()
    res = calibrator(proj)(proj.offsets)
    d = W.diagnose(proj, res.to_calibration(), A, B)
    json.dumps(d, allow_nan=False)
    pe = W.prediction_error(proj, res.to_calibration(), A, B)
    json.dumps(pe, allow_nan=False)
    out = W.refine(proj, A, B, calibrator(proj))
    json.dumps(W.json_safe({k: v for k, v in out.items() if not k.startswith("res_")}), allow_nan=False)


if failures:
    print(f"verify_wand_sync FAILED: {failures}")
    sys.exit(1)
print("verify_wand_sync PASSED")
