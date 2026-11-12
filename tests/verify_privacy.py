"""What the files and texts a user hands to other people say about their computer (G243-G247).

- the MATLAB export names the video by its file name, not its folder;
- a lens profile (handed from person to person) does not carry the board video's folder;
- a project whose videos are inside its folder stores no full path, still opens, and still finds them after
  the folder moved; a video elsewhere keeps its full path (it is how it is found again);
- the Error Report text hides the user folder, the user name and anything shaped like an access token;
- SAM 3D Body's backbone is loaded from the local folder without asking github.com;
- Help > About says when Kinetrace goes online.

No GPU, no window. Run: .venv\\Scripts\\python.exe tests\\verify_privacy.py
"""
import json
import os
import shutil
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

OUT = os.path.join(ROOT, "tests", "out", "privacy")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
FAILED = []


def check(ok, what, detail=""):
    line = ("OK   " if ok else "FAIL ") + what + (f"  [{detail}]" if detail and not ok else "")
    print(line.encode("ascii", "backslashreplace").decode("ascii"))
    if not ok:
        FAILED.append(what)


def text_of(v):
    return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)


from kinetrace import projectfile as pf  # noqa: E402
from kinetrace.project import Project  # noqa: E402
from kinetrace.session import TrackingSession  # noqa: E402

# ---- the MATLAB export ------------------------------------------------------------------------
from scipy.io import loadmat  # noqa: E402

secret = os.path.join(OUT, "Jane Doe", "field season")
os.makedirs(secret, exist_ok=True)
s = TrackingSession(os.path.join(secret, "clip01.mp4"), 12, 30.0, 640, 480)
s.add_point(0, 10.0, 20.0, name="snout")
mat = os.path.join(OUT, "clip01.mat")
s.export_mat(mat)
got = str(np.ravel(loadmat(mat)["video_path"])[0])
check(got == "clip01.mp4", "the MATLAB export names the video by its file name only", got)

# ---- lens profiles ------------------------------------------------------------------------------
from kinetrace import lens  # noqa: E402

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
prof = lens.LensProfile(640, 480, K, np.zeros(5), rms=0.3, n_views=20, source="board.mp4",
                        report={"verdict": "good"})
prof.board = {"video": "board.mp4", "path": os.path.join(secret, "board.mp4"), "size": [640, 480]}
js = prof.to_json()
check("path" not in (js.get("board") or {}) and js["board"].get("video") == "board.mp4",
      "a saved lens profile keeps the board video's name and drops its folder", str(js.get("board")))
check("Jane Doe" not in json.dumps(js), "... nothing of the folder is left anywhere in the file")
scan = types.SimpleNamespace(video=os.path.join(secret, "board.mp4"), frames=[3, 9], size=(640, 480), rotation=0,
                             gopro_orientation="", board=None)
rec = lens.board_record(scan, [0, 1], (9, 6), 0.024)
check("path" not in rec and rec["video"] == "board.mp4" and rec["frames"] == [3, 9],
      "a new checkerboard record has no full path", str(rec))
bare = lens.LensProfile(640, 480, K, np.zeros(5), rms=0.3, n_views=20, source="board.mp4", report={})
bare.rotation = None
named, _said = lens.record_turn(bare, os.path.join(secret, "board.mp4"), (640, 480), 0)
check(named is not None and "path" not in named.board and named.board["video"] == "board.mp4",
      "a turn read later from the board video records no full path", str(getattr(named, "board", None)))
for mod in ("lens.py", "charucowizard.py"):
    src = open(os.path.join(ROOT, "kinetrace", mod), encoding="utf-8").read()
    check('"path": os.path.abspath' not in src, f"{mod} builds no board record with a full path")

# ---- the project file ----------------------------------------------------------------------------
proj = os.path.join(OUT, "study.kinetrace")
inside = os.path.join(proj, "videos", "camA.mp4")
outside = os.path.join(OUT, "elsewhere", "camB.mp4")
os.makedirs(os.path.dirname(outside), exist_ok=True)
for f in (outside, os.path.join(OUT, "elsewhere", "camA.mp4")):
    with open(f, "wb") as fh:
        fh.write(b"not really a video")


def session(path):
    x = TrackingSession(path, 20, 30.0, 320, 240)
    x.add_point(0, 5.0, 6.0, name="snout")
    return x


first = os.path.join(OUT, "elsewhere", "camA.mp4")
p = Project([session(first), session(outside)], ["camA", "camB"], [0.0, 1.5], 0, [1.0, 1.0])
pf.save(p, proj)                                     # the project exists; then camA's video is put inside it
os.makedirs(os.path.dirname(inside), exist_ok=True)
shutil.copyfile(outside, inside)
p.sessions[0].video_path = inside
pf.save(p, proj)
cams = json.load(open(os.path.join(proj, "project.json"), encoding="utf-8"))["cameras"]
va, vb = cams[0]["video"], cams[1]["video"]
check("path" not in va and va.get("relative_path") == "videos/camA.mp4",
      "a video inside the project folder is stored by its place in the folder only", str(va))
check(os.path.normcase(vb.get("path", "")) == os.path.normcase(outside),
      "a video elsewhere keeps its full path (how it is found again)", str(vb))
whole = "".join(open(os.path.join(proj, n), encoding="utf-8").read() for n in ("project.json", "kinetrace.json"))
check(whole.count("camA.mp4") == 1, "... and camA's folder appears nowhere else in the project's index files")
q = pf.load(proj)
check(os.path.isfile(q.sessions[0].video_path) and os.path.samefile(q.sessions[0].video_path, inside),
      "the project opens with camA's video where it is", q.sessions[0].video_path)
check(os.path.samefile(q.sessions[1].video_path, outside), "... and camB's", q.sessions[1].video_path)
check(np.array_equal(q.sessions[0].tracks, p.sessions[0].tracks, equal_nan=True)
      and list(q.offsets) == [0.0, 1.5], "... with the same points and offsets")
meta = pf.read_meta(proj)
base = pf.video_base(proj, meta)
check(pf.locate_video(cams[0], base) is not None, "the app's video search finds camA from the stored place")
readme = open(os.path.join(proj, "README.txt"), encoding="utf-8").read()
check("full path" in readme.lower() or "full folder path" in readme.lower(),
      "README.txt says when a full path is stored and how to share without it")
moved = os.path.join(OUT, "moved", "study.kinetrace")
os.makedirs(os.path.dirname(moved), exist_ok=True)
shutil.move(proj, moved)
q = pf.load(moved)
want = os.path.join(moved, "videos", "camA.mp4")
check(os.path.isfile(q.sessions[0].video_path) and os.path.samefile(q.sessions[0].video_path, want),
      "after the whole folder moved, camA's video is the one inside it", q.sessions[0].video_path)
pf.save(q, moved)                                    # a save from the opened project keeps the rule
va2 = json.load(open(os.path.join(moved, "project.json"), encoding="utf-8"))["cameras"][0]["video"]
check("path" not in va2 and va2.get("relative_path") == "videos/camA.mp4", "saved again: the same", str(va2))
# a recovery copy lives elsewhere (by project id): it must keep the full path
fz = pf.freeze(q, {"tools": {}}, pf.new_id(), target=moved, binary_tracks=True)
rc = json.loads(text_of(fz.files["project.json"]))["cameras"][0]["video"]
check(bool(rc.get("path")), "a recovery copy keeps the full path", str(rc))
one = os.path.join(OUT, "moved", "single.kinetrace.zip")
fz = pf.freeze(q, {"tools": {}}, pf.new_id(), target=one, layout="zip")
zc = json.loads(text_of(fz.files["project.json"]))["cameras"][0]["video"]
check(bool(zc.get("path")), "the single-file form keeps the full path (the video is not inside it)", str(zc))

# ---- the Error Report ------------------------------------------------------------------------------
from kinetrace import crashlog  # noqa: E402

redact = getattr(crashlog, "redact", None)
check(redact is not None, "crashlog.redact exists")
if redact is not None:
    win_home, user = "C:\\Users\\someone", "someone"
    t = redact('File "C:\\Users\\someone\\Kinetrace\\kinetrace\\app.py", line 5', home=win_home, user=user)
    check("someone" not in t and "~\\Kinetrace\\kinetrace\\app.py" in t, "the user folder reads ~", t)
    t = redact("opened c:/users/SOMEONE/Videos/x.mp4", home=win_home, user=user)
    check("someone" not in t.lower() and "~/Videos/x.mp4" in t, "... also written with / and in another case", t)
    t = redact("C:\\Users\\someoneelse\\x", home=win_home, user=user)
    check("someoneelse" in t, "another user's folder that starts with the same letters is left alone", t)
    t = redact("/media/janedoe/disk/x.mp4 and /home/janedoe", home="/home/janedoe", user="janedoe")
    check("janedoe" not in t and "<user>" in t and "~" in t, "the user name elsewhere reads <user>", t)
    t = redact("janedoesnot and X", home="/home/janedoe", user="janedoe")
    check("janedoesnot" in t, "a longer word that starts with the name is left alone", t)
    t = redact("al was here; /home/al/x", home="/home/al", user="al")
    check("al was here" in t and "~/x" in t, "a two-letter user name is hidden only as the user folder", t)
    t = redact("token hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345 sent", home="/home/x1", user="x1")
    check("AbCdEf" not in t and "hf_<hidden>" in t, "anything shaped like a Hugging Face token is hidden", t)
    logs = os.path.join(OUT, "Jane Doe", "logs")
    os.makedirs(logs, exist_ok=True)
    with open(os.path.join(logs, crashlog.LOG_NAME), "w", encoding="utf-8") as fh:
        fh.write("2026-01-01 10:00:00.000  ERROR  could not read " + os.path.join(secret, "clip01.mp4") + "\n")
    os.environ[crashlog.ENV] = logs
    crashlog._identity = lambda: (os.path.join(OUT, "Jane Doe"), "Jane Doe")
    rep = crashlog.report_text()
    check("Jane Doe" not in rep and "clip01.mp4" in rep, "the Error Report text has the user folder hidden",
          rep[:300])
    del os.environ[crashlog.ENV]

# ---- the install log ----------------------------------------------------------------------------------
sys.path.insert(0, ROOT)
import install  # noqa: E402

old = install.LOG
install.LOG = os.path.join(ROOT, "logs", "install-test.log")
ask = install.where_to_ask()
install.LOG = old
check("user name" in ask, "asking for the install log says it holds folder names (user name)", ask)

# ---- SAM 3D Body's backbone, without github.com -------------------------------------------------------
from kinetrace import bodypose  # noqa: E402

local_hub = getattr(bodypose, "local_hub", None)
check(local_hub is not None, "bodypose.local_hub exists")
if local_hub is not None:
    calls = []

    def fake_load(repo, model, *a, **k):
        calls.append((repo, model, a, dict(k)))
        return "net"

    hub = types.SimpleNamespace(load=fake_load)
    have = os.path.join(OUT, "facebookresearch_dinov3_main")
    os.makedirs(have, exist_ok=True)
    with open(os.path.join(have, "hubconf.py"), "w", encoding="utf-8") as fh:
        fh.write("# test\n")
    with local_hub(hub, folder=have):
        r = hub.load("facebookresearch/dinov3", "dinov3_vith16plus", source="github", pretrained=False,
                     drop_path=0.1)
        hub.load("someone/else", "m", source="github")
    c = calls[0]
    check(r == "net" and c[0] == have and c[3].get("source") == "local" and c[1] == "dinov3_vith16plus"
          and c[3].get("pretrained") is False and c[3].get("drop_path") == 0.1,
          "the backbone is built from the local folder with the same arguments", str(c))
    check(calls[1][0] == "someone/else" and calls[1][3].get("source") == "github",
          "another repository's load is passed through untouched", str(calls[1]))
    check(hub.load is fake_load, "torch.hub.load is put back afterwards")
    calls.clear()
    with local_hub(hub, folder=os.path.join(OUT, "not-there")):
        hub.load("facebookresearch/dinov3", "m", source="github")
    check(calls[0][0] == "facebookresearch/dinov3" and calls[0][3].get("source") == "github",
          "without the local folder the call is unchanged", str(calls[0]))
    try:
        with local_hub(hub, folder=have):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    check(hub.load is fake_load, "... and it is put back after an error too")

# ---- Help > About ------------------------------------------------------------------------------------
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from kinetrace import updatedialog  # noqa: E402

about = updatedialog.about_html()
check("goes online only" in about and "Hugging Face" in about and "GitHub" in about
      and "never sent" in about, "Help > About says when Kinetrace goes online and that nothing is sent",
      about[-400:])

if FAILED:
    print(f"PRIVACY FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("PRIVACY PASSED")
