# Kinetrace — User Manual

**For people who have never done motion tracking before.** No experience with
tracking software, programming, or computer vision is assumed. If a word looks
technical, it is explained the first time it appears, and again in the
[Glossary](#16-glossary) at the end.

Read sections 1–7 before your first real video. The rest you can look up when you
need it.

---

## Contents

1. [What this program does](#1-what-this-program-does)
2. [What you need before you start](#2-what-you-need-before-you-start)
3. [Starting the program](#3-starting-the-program)
4. [The five ideas you need](#4-the-five-ideas-you-need)
5. [A tour of the window](#5-a-tour-of-the-window)
6. [Your first tracking session](#6-your-first-tracking-session)
7. [Checking the result and fixing mistakes](#7-checking-the-result-and-fixing-mistakes)
8. [Reading the timeline](#8-reading-the-timeline)
9. [Marking events](#9-marking-events)
10. [Filming with several cameras](#10-filming-with-several-cameras)
11. [Saving and coming back later](#11-saving-and-coming-back-later)
12. [Getting your numbers out](#12-getting-your-numbers-out)
13. [Measuring a person's joints and joint angles](#13-measuring-a-persons-joints-and-joint-angles)
14. [Choosing the settings that matter](#14-choosing-the-settings-that-matter)
15. [When something goes wrong](#15-when-something-goes-wrong)
16. [Glossary](#16-glossary)
17. [Keyboard and mouse reference](#17-keyboard-and-mouse-reference)

---

## 1. What this program does

You have a video of an animal moving. You want **numbers** out of it: where was
the snout in each picture of that video? Where was the left hind foot? How did
the tail tip move over time?

Doing that by hand means opening every single picture and clicking the snout —
thousands of times. Kinetrace does the clicking for you. **You show it where
something is once, and it follows that thing through the rest of the video.**

What you get at the end is a table. One row per picture, one pair of columns per
body part, holding the position of that body part in that picture. You can open
that table in Excel, MATLAB, R, Python — whatever you already use.

**What it is not.** It does not identify species, count animals, or interpret
behaviour. It answers exactly one question, very precisely: *where was this
thing, in this picture?* Everything else is your analysis, downstream.

**You stay in charge.** The program never runs off on its own. You watch it work
picture by picture, you can stop it at any moment, and wherever it gets something
wrong you click the right place and carry on. Your corrections always win.

---

## 2. What you need before you start

**A video file.** Almost any common format works (`.mp4`, `.mov`, `.avi`,
`.mkv`, and others). It can be long — tens of thousands of pictures — and it can
be high resolution, including 4K. The program never loads the whole video into
memory, so size is not a problem.

**One thing to check about your video.** Your camera must have recorded at a
*constant* rate — the same number of pictures every second, start to finish.
Most cameras do. Some phones and screen recorders do not, and then picture
numbers no longer line up with time. Kinetrace checks this when you open a file
and warns you if it looks wrong, and it tells you the exact command to fix the
file. Do not ignore that warning: the numbers it produces would not be
trustworthy.

**The frame rate comes from the file.** High-speed footage is fine: rates up to
100,000 frames per second are taken as the file states them. If a file does not
state a usable rate, Kinetrace measures it from the frames' own time stamps, or,
when even those give nothing, assumes 30 frames per second — and says so in a
notice over the video every time you open it. Check that number against the
rate your camera recorded at: every time, speed and camera sync depends on it.
**If the file's number is wrong**, and high-speed cameras often save footage for
slow-motion playback, so the file says 30 while the camera filmed at 240 or
1000, click the **fps button** under the camera's offset in the CAMERAS panel
(it reads, say, *30 fps*) and enter the rate set on the camera. A rate set by
hand shows a star (*240 fps \**). Enter the file's number again to undo it.
(If a file claims more frames than can actually be read, Kinetrace uses the
frames that exist and tells you.)

**A Windows or Linux computer with an NVIDIA graphics card, or a Mac with
Apple Silicon (M1 or later)**, is strongly recommended. It works without a
graphics card, just much more slowly (a long 4K video becomes an overnight
job), and one feature — the 3D human body model, SAM 3D Body — is switched off
without an NVIDIA card, because its maker's code runs only there; the program
greys it out and offers the 2D model instead. Linux needs Ubuntu 22.04 or newer
(or another distribution of that age); a Mac needs macOS 14 or newer — Intel
Macs cannot run it. 16 GB of memory is comfortable; with less, the program uses
a smaller working picture for one of its models and says so. **Help → System
Check…** lists what your computer has and, feature by feature, what runs on
it, what runs slower and what is off; the status bar along the bottom of the
window always shows which processor is in use.

**Nothing else — not even Python.** Everything the program needs lives inside
its own folder; the first start fetches whatever is missing, without asking
you anything. It does not install anything into your operating system.
Deleting the folder removes it completely; your projects and exports are saved
where you choose and stay. **Help → Kinetrace's Folders…** lists every place
the program uses, and `docs/INSTALL.md` in the folder has the full installation
steps and how to uninstall.

---

## 3. Starting the program

**Windows:** double-click `run.bat` in the program folder. (If Windows says
*"Windows protected your PC"*, click *More info*, then *Run anyway*.)

**Mac:** double-click `Kinetrace.command` in the program folder. The first
time, the Mac refuses it because it came from the internet and is not signed
by Apple: click **Done**, not *Move to Trash*. On macOS 15 or newer, open
**System Settings → Privacy & Security**, scroll down to the line about
`Kinetrace.command`, click **Open Anyway** and confirm, then double-click it
again and click **Open**. On macOS 14: right-click the file, choose **Open**,
and confirm. A Terminal window opens: leave it open until the program's window
appears (if the Mac asks whether Terminal may access your Documents, Desktop
or Downloads folder, allow it). That first run makes **`Kinetrace.app`** in the
folder: from then on, double-click that instead — no Terminal window, and you
can drag it to the Dock. Put the program folder in your home folder rather than
on an iCloud-synced Desktop or Documents.

**Ubuntu:** double-click `run.sh` in the program folder and choose *Run in
Terminal* — or open a terminal in the folder and type `./run.sh`. If the window
needs a few system pieces that are not installed, the program installs them
for you; that is the one step that asks for your password (if it cannot ask,
it prints the exact `sudo apt-get install …` line to run yourself). Ubuntu's
own `python3` often comes without the `venv` part; the launcher notices and
downloads a private Python of its own instead of failing.

The program works out by itself which kind of computer it is on — NVIDIA
graphics card or not, Mac or PC, Python already there or not — and fetches the
matching version of its supporting software. Nothing is asked.

**The very first launch takes a long time** — it downloads up to about 4 GB of
supporting software into its own folder (about 1.5 GB on a Mac) (and, if the computer has no Python of
its own, a private copy of that too, about 20 MB). That happens once, and it
ends by printing a **system check**: what it found and what your computer can
run. If the connection drops, just start it again — it keeps what it already
installed and carries on from there. (To redo the installation of the
packages from scratch, run `python install.py --force` with the Python inside
the program's `.venv`.) Later launches start in a few seconds.
The first time you press the **Track** button, and the first time you click on
an animal, it downloads one more piece each (a tracking model of 66 MB, a
segmentation model of 617 MB): a window says what is downloading, how far it
has got and about how long is left, and **Cancel** stops it — the next try
carries on where it stopped. A connection that is cut half way through resumes
the same way, and a disk that is full is reported as a full disk (not as a
network problem). Every piece is checked before it is used (a file
that is not exactly the expected one is thrown away, never run). After that you
never need an internet connection again, except to get a newer version when you
ask for one.

If it ever fails to start, see [section 15](#15-when-something-goes-wrong).

### Optional: the two models that need Meta's permission

Everything above downloads by itself. Two **optional** models are released by
Meta only to people who ask, so the program cannot fetch them for you. You do
not need either to track animals: without them the program uses **SAM 2.1** for
outlines and **ViTPose** for human joints.

**SAM 3** — the best outlines (Segment, **S**), about half the speed, 3.4 GB:

1. Make a free account at `huggingface.co`, open
   `huggingface.co/facebook/sam3`, and press the button to request access.
   Wait for the e-mail saying you were granted it (often minutes, sometimes
   days).
2. In your Hugging Face account: *Settings → Access Tokens → Create new token*,
   type **Read**, and copy it.
3. In Kinetrace: **Ctrl+,** (Settings, also the last entry of the **Segment ▾**
   menu) → paste the token → **Save token**. It is kept inside the program's
   folder (`models/hf/token`) and sent only to Hugging Face.
4. Choose **SAM 3** in the **Segment ▾** menu. The first outline downloads it
   (3.4 GB, with a progress window); after that it works offline.

Or, without a token in the program: download it on a computer that has access
and put it in the program's folder as `models/sam3/` — the exact files and a
copy-paste command are in `models/sam3/PUT_FILES_HERE.txt` and in
`docs/INSTALL.md`. Leave out `sam3.pt`: it is a second 3.45 GB copy the program
does not use. A copy there is used first, and SAM 3 becomes the default.

**SAM 3D Body** — joints of a person in 3D from one camera (section 13): ask for
access at `huggingface.co/facebook/sam-3d-body-dinov3` the same way, then
follow the three steps in section 13, *The two models, and which to pick*. It
needs an NVIDIA graphics card: **on a Mac, or a computer without one, skip it**.

**Body → Find People & Measure Joints…** and the **Segment ▾** menu say for each model whether it is
ready, needs downloading, or still needs access or a token.

### Getting a newer version

Open **Help → Check for Updates…**. The program asks GitHub whether a newer
version has been published. It sends nothing about you or your computer, and
it never checks unless you ask. If there is one, you see what is new, and
**Update now** installs it (a download cut off by a lost connection resumes where it
stopped, and a full disk is named as such). Only the program's own files are replaced: your
projects, downloaded models, unsaved-work copies, saved skeletons and settings
are kept. **Restart now** then closes the program (asking first if you have
unsaved work) and starts the new version. If the new version needs extra
supporting software, that first start installs it, which takes a few minutes
and needs the internet.

If the program will not start at all, update it with the program closed:
double-click `update.bat` (Windows) or `Update.command` (Mac), or type
`bash update.sh` in a terminal in the program folder (Ubuntu). Then start it as
usual.

**Help → About Kinetrace** shows which version you have, the licence (free for
any non-commercial use, such as research, teaching and study) and the models
the program builds on. Kinetrace is developed at biomechLab@CMC.

---

## 4. The five ideas you need

These five ideas are all the background required. Everything in the program is
built out of them.

### 4.1 A video is a stack of still pictures

A video is a flip-book. Each still picture is called a **frame**. They are
numbered starting at **0** — so a 600-frame video runs from frame 0 to frame
599. Every number this program produces is attached to a frame number.

If your camera recorded 30 frames per second, frame 90 is 3 seconds in. If it
recorded 300 frames per second, frame 90 is 0.3 seconds in. Hover the mouse
over the timeline and the program shows you both the frame number and the time.

### 4.2 Tracking a point

You click once on something in one frame — say the tip of the snout. That click
creates a **point**. The program then looks at the small patch of picture around
your click and searches for that same patch in the next frame, and the next, and
so on. That is **tracking**.

This works well when the thing you clicked *looks like something* — a dark eye, a
paint mark, a claw against pale sand, a distinctly coloured scale. It works badly
on smooth, featureless surfaces, because every part of them looks the same. A
plain grey tail is the classic hard case: there is nothing to recognise, so the
tracker slides along it and drifts away.

That limitation is why the next idea exists.

### 4.3 The silhouette

Instead of following a small patch, the program can outline the **whole animal**
— its complete shape against the background, in every frame. That outline is
called the **silhouette**. The tool that draws it is called **Segment** (key
**S**), and what it outlines is an **animal**: a row in the LAYERS panel that
holds the animal's points. An animal does not need a silhouette — it can be
just a group of points — but once it has one, its body parts can be found from
it.

The silhouette is powerful because some body parts are *defined by shape* rather
than by appearance:

- the **tail tip** is simply the far end of the outline,
- the **midline** is the line running down the centre of the body,
- the **feet and wing tips** are the parts that stick out from the body.

None of those need a recognisable patch of texture. They can be computed from the
shape alone. **This is how Kinetrace solves the plain-grey-tail problem** — and
it is the single most important thing to understand about using it well.

So there are two different ways a body part can be found:

| Way | How it works | Good for |
|---|---|---|
| **By appearance** (you click it) | follows a recognisable patch | eyes, markings, claws, joints you can see |
| **From the silhouette** (computed) | reads the animal's outline | tail tips, midline, feet, wing tips |

You do not have to choose blindly. The program picks sensible defaults for you,
and shows you which is which.

### 4.4 Landmarks and skeletons

A **landmark** is a named body part you want measured — "snout", "left hind
foot", "tail tip". Naming matters: the names become the column headings in your
results, and if you use several cameras, matching names are what tie them
together.

A **skeleton** is a ready-made list of landmarks for a kind of animal, so you do
not have to invent and type them. Kinetrace ships with six:

- Lizard / iguana (14 landmarks)
- Quadruped, generic (10)
- Gliding mammal, e.g. flying squirrel (12)
- Flying lizard, *Draco* (12)
- Undulating body — fish, eel, anything that swims by bending (5)
- Bird / bat, with wings (8)

Pick the closest one. In each skeleton, some landmarks are marked as coming
**from the silhouette** — those you never click, they appear on their own once
you have outlined the animal. The rest you place yourself with a click.

> **Left and right from the silhouette assume a view from above.** Feet and wing
> tips found from the outline (`foot_FL`, `wing_tip_L` and the like: F = fore,
> H = hind, L = left, R = right) are named as the animal's own left and right
> seen from **above**, with its back towards the camera. Filmed from **below**,
> left and right come out swapped: right-click each of those landmarks →
> **Data source** and pick the other side's rule (for `foot_FL`, *From
> silhouette: fore-right extremity (foot)*), so each name stays true. Tables
> exported before 22 September 2026 from footage filmed from above had them
> the wrong way round.

You can also build your own list: **Skeleton ▾ → Custom skeleton…** asks for a
name, the head landmark, the landmarks (one per line) and, if you like, the
bones to draw between them (one `a - b` pair per line). To have a landmark
computed from the silhouette instead of clicked, add a rule after its name:
`tail_tip = tip` (the far end of the body), `mid = midline:0.5` (a fraction of
the way along the body, from 0 at the head to 1 at the tail tip — a fraction,
not a percentage), `body = centroid` (the middle of the outline),
`wing_L = ext:L` or `ext:R` (the part sticking out farthest on each side), or
`foot_FL = ext:FL` (also `FR`, `HL`, `HR`: the fore and hind feet on each
side). A rule the program cannot use is reported when you press OK, and that
landmark is tracked by appearance instead. Your skeleton is saved in the
program's `skeletons` folder and stays in the menu from then on.

A skeleton always belongs to one **animal**: its landmarks are named after it
("squirrel snout", shown as just "snout" under the squirrel in LAYERS), so two
animals in one video can use the same skeleton. Once an animal's points and
bones are as you want them, right-click the animal → **Save its points and
bones as a skeleton template…** to reuse them on another animal.

### 4.5 Confidence

For every body part in every frame, the program records how sure it is — a
number called **confidence**. High means "I can clearly see what I am
following." Low means "I have lost it."

Confidence is not the same as *visible*. If a rock passes in front of the snout,
the snout is hidden but the program still knows roughly where it is and stays
confident. If the snout leaves the picture entirely, or the tracker slides onto
the background, confidence collapses.

You never have to read these numbers. The program uses them for you: it colours
uncertain stretches **red** on the timeline so you can see at a glance where to
look, and it can stop automatically when it loses something.

---

## 5. A tour of the window

```
 File  Edit  View  Skeleton  Events  3D  Body  Help      ← the menus
┌────────────────────────────────────────────────────┬──────────────────────┐
│ 1 Open video ✓ › 2 Segment › 3 Skeleton › 4 Track  │ Layers               │
├────────────────────────────────────────────────────┤ CAMERAS  + Add video │
│                                                    │ cam1 (reference)     │
│                                                    │ offset 0.000  ◂ ▸    │
│                    the video                       │                      │
│                (click things here)                 │ LAYERS               │
│                                                    │ + Animal  + Point    │
│                                          ⊖ ⊕ ⤢     │ Skeleton ▾  Delete   │
├────────────────────────────────────────────────────┤ ● iguana             │
│ ⊖⊕⤢           the timeline                         │    ■ snout           │
│        one horizontal bar per body part            │    ■ eye             │
├────────────────────────────────────────────────────┤ Scene                │
│ 120 / 599  ⏴ ▶ ⏵   tools and toggles…   Track ▶    │    ■ rock mark       │
└────────────────────────────────────────────────────┴──────────────────────┘
 frame 120 / 599   last tracked: 450   cuda: …      ← the status bar
```

**The menus along the top:** **File** (open a video or a project, save,
export), **Edit** (undo, notes, the hidden mark, the name recorded on your
notes), **View** (the panel, this strip, trails, the loupe, display filters),
**Skeleton** (ready-made landmark lists for an animal), **Events**, **3D** (several cameras:
sync, calibration, 3D positions), **Body** (people's joints and joint angles)
and **Help** (this manual, **F1**, and the keyboard reference). **Hover the
mouse over any menu entry** and a line appears saying what it does. In the 3D
menu, the entries that need something you have not done yet — a second camera,
a calibration — mostly stay clickable and tell you what is missing when you
click them.

**The strip under the menus** is a four-step checklist: *1 Open video → 2
Segment the animal → 3 Skeleton / landmarks → 4 Track*. Steps 2 and 3 are
marked *optional*, because they are. Finished steps get a tick, and the step
outlined as next is **Track** as soon as a video is open. Click a step to do
it: 1 opens a video, 2 switches on the Segment tool, 3 opens the list of
skeletons, and 4 **starts tracking**. Text to the right of the four steps,
after the dividing line and marked ⓘ, is a hint about what to do next — it is
advice, not a fifth step. The strip hides itself once something has been
tracked; you can also close it with **×**, and **View → Getting started
strip** brings it back.

**The middle** is your video. This is where you click on things. You can zoom
with the mouse wheel (or **+** and **−**, or the zoom-out / zoom-in buttons in
the picture's lower-right corner) and pan by holding the left button and
moving the mouse (the middle button works too, and so does the **Pan** tool,
key **H**). A plain left **click** on the video places the point selected in
LAYERS; a short right click on a marker clears that point on that frame.
Markers are never dragged, and a sideways swipe on a touchpad does not zoom.
Press **R** (or the fit button in that corner) at any time to fit the whole
picture back in the window. With
several cameras the videos share this space as a grid; the one you are working
in is outlined, and clicking another switches to it (section 10). Short notices
appear over the video for a few seconds when something needs saying.

**The timeline** underneath is explained fully in [section 8](#8-reading-the-timeline).
It is also your scrubber: **click or drag anywhere along it to move through the
video.** There is no separate slider. Drag the boundary between the video and
the timeline to give the timeline more rows, and hover over it to see the frame
number and the time in seconds. The three magnifier buttons in the timeline's
top-left corner, left of the frame numbers, zoom the timeline out, in, and back
to the whole video.

**The control bar at the bottom**, from left to right:

- the **frame box** (type a frame number and press Enter), then **⏴ ▶ ⏵** —
  back one frame, play / pause a preview (display only: nothing is tracked),
  forward one frame;
- **±10**, the jump size for **Shift+F** / **Shift+B**, and **● 3px**, the size
  the markers are drawn at;
- the tools: **Point** (**N**; its **▾** picks the region shape or a ball
  marker), **Segment** (**S**; its **▾** picks the outlining model and ends with
  *Settings…*) and **Pan** (**H**). Only one of these three is on at a time:
  picking one switches the other two off;
- four switches, explained in [section 14](#14-choosing-the-settings-that-matter):
  **Follow** (keeps your points in view while zoomed in; off until you switch
  it on), **Mask** (shows the silhouettes; off hides every one),
  **Auto-pause** and **ROI**;
- the blue **Track ▶** button that starts the work. Its **▾** arrow chooses
  automatic or semi-automatic tracking and the point model.

Until a video is open most of the bar is greyed out. The window opens as large
as your screen allows. On a narrower window the tools and switches give up
their names **one at a time**, the least-used first (Pan, then ROI, Mask,
Follow, Auto-pause, Segment, and Point last), and show **only their
icons** — hover over one to see its name and what it does, and widen the window
to bring the names back. The Track button always keeps its words.

**The right panel** (titled *Layers*) has two sections:

- **CAMERAS** — one compact line per camera, so ten or fifteen fit at once:
  **▶** (open the row), the **eye** (show / hide this camera's view), its
  **number** and name (camera 1 is marked **(reference)**) and its offset. The
  number is the camera's place in the **camera order** — the order of
  calibrations, 3D and exports (*Camera order…* under the list changes it).
  **▶** opens the row: **Align here** and **×** (remove this camera) on its
  first line, its **offset** — the frame this camera shows when the reference
  camera is at its frame 0 — with **◂ ▸** to nudge it a frame at a time (and
  **×2** for a camera filming at twice the reference rate), then its frame rate
  and a count of its points and tracked frames. With one video there is a
  single row and nothing to set; **＋ Add video** adds another camera of the
  same event (section 10).
- **LAYERS** — everything you track, as a tree. Each **animal** (one
  individual animal, or any object you follow) is a row with its points
  underneath; then a **Scene** row holds the points that belong to no animal —
  wand ends, a reference mark on a rock, calibration points. (With no animal
  at all, the points are simply a list, with no Scene row.) An animal's swatch
  is filled when it has a silhouette and an outline when it has none yet; a
  small dot inside means its points are kept on its silhouette (section 14).
  Each point shows a small symbol saying how it is found (Step 4 below) and
  **AT**, **CT** or **MS**: the tracker that follows it — AllTracker,
  CoTracker3 or Moving spot (right-click → *Tracker* changes it).

  Above the tree: **＋ Animal** (a new animal, with no silhouette yet),
  **＋ Point** (a new point with no position yet, in the selected animal, else in
  Scene), **Skeleton ▾** (a ready-made list of points for the selected animal)
  and **Delete** (deletes what is selected). Under the tree, a line gives the
  selected animal's numbers: its points, clicks and silhouette frames.

  **Checkboxes:** a point's checkbox shows / hides that point; an animal's
  checkbox shows / hides *its* silhouette; Scene's checkbox shows / hides the
  Scene points. **Selecting** works like a list of files: a plain click selects
  only that row; **Ctrl+click** or **Shift+click** adds rows; **Ctrl+A**
  selects every animal and every point. **Track tracks exactly what is
  selected here**: a point row is that point, an animal row is its silhouette
  and all its points. **Drag** points onto another animal (or onto Scene) to
  move them there.

  **Names:** double-click a row to rename it; right-click it for everything
  else (on the video, hold the right button on a marker instead: a short right
  click there clears it on that frame). A point of an animal is called
  "*animal part*" everywhere — in exports, in 3D and in the other cameras — for
  example *squirrel snout*; the tree shows only the part (*snout*), and
  renaming it in the tree changes the part. Renaming an animal renames its
  points in every camera. A name may not start with **= + -**
  or **@**: a spreadsheet opening an export would run it as a formula, so such
  a name is refused (and taken off a name that arrives in a file). Renaming a point
  in capital letters only (`snout` → `Snout`) works, and the next save keeps its
  data.

Hide the panel with **Ctrl+1** (**View → Layers panel**) if you want
more room.

**The status bar along the very bottom** of the window reports what just
happened, the frame you are on and the last frame with tracking in it
(*frame 120 / 599   last tracked: 450*), a coloured badge saying which
processor the models run on — green **● GPU** with the graphics card's name,
or amber **● CPU**, which means the processor is doing the work and tracking
will be slower. The program measures both once on the first start and uses
whichever is faster, so the badge is amber only when there is no usable
graphics card (or, rarely, when the card measured slower). Hover over the
badge for the measured times and what that means, or open **Help → System
Check…**. While tracking, the speed readout starts with GPU or CPU and ends
with an estimated finish time.

---

## 6. Your first tracking session

Work through this once with a short, easy clip before using a clip that matters.

### Step 1 — Open the video

**File → Open Video…** (or **Ctrl+O**). Pick your file. If the video or
project already open has unsaved changes, you are asked first, as when you
close the program: **Save**, **Discard** (drop them) or **Cancel** (open
nothing). Open Project, Open Folder of Videos and Recover Unsaved Work ask the
same.

While it opens, a card in the middle of the window says what is happening —
reading the file, checking its frame rate, checking that every frame can be read
— with a bar that fills as it goes. A 4K file takes a few seconds; a file on a
network drive longer (the card says so, and a copy on your own disk opens and
plays much faster). **Cancel** (or **Esc**) stops it without changing anything
that was already open. The same card appears when you open a project or add
cameras; several cameras are read at the same time. While it is up it holds
the mouse and keyboard in every window of the program (a floated panel, the
side-by-side view), not only the main one.

The first frame appears. If a warning about "variable frame rate" appears, stop
and fix the file first — see [section 2](#2-what-you-need-before-you-start).
If you worked on this video before without saving a project, the program asks
**Restore unsaved work?**: **Yes** carries on exactly where you stopped (see
[section 11](#11-saving-and-coming-back-later)).

### Step 2 — Go to a good starting frame

Tracking always begins **at the frame you are looking at** and moves forward.
There is no backwards tracking.

So find a frame where everything you care about is clearly visible and not hidden
behind anything. That is usually not frame 0. Move around with:

- the **timeline** — click or drag anywhere along it,
- **F** and **B** — forward and back one frame,
- the **frame box** at the bottom left — type a number and press Enter.

Everything you do next happens on this frame.

### Step 3 — Outline the animal

Press **S** (or click the **Segment** button). The cursor changes.

**Click once on the animal.** Within a second or so its outline appears, filled
with a translucent colour. The very first time you do this the program loads
its outlining software, which takes a little longer. Your first click also
makes the animal's row in LAYERS (named *animal*; double-click it to give it a
better name, such as *iguana*).

**S** always outlines the animal selected in LAYERS. With only one animal and
nothing selected, it outlines that one. With several animals, select the one
you want first (or press **＋ Animal** for a new one); otherwise the click does
nothing and a notice says so.

Check the outline actually follows the animal:

- **It grabbed too much** (the animal plus a rock, or plus your hand):
  **Shift+click** on the part that should *not* be included. The outline shrinks.
- **It missed part of the animal** (the tail is not included): plain-click on the
  missing part. The outline grows.
- **It picked something else entirely:** drag a box around the animal instead.
- **You clicked in the wrong place:** right-click your click marker and choose
  **Remove this click**.

Repeat until the outline matches the animal. It is worth getting this right —
several landmarks are computed from this shape. For now only the frame you are
on is outlined; the rest of the video follows when you press **Track**.

Press **S** or **Esc** when you are done. (The animal's row in LAYERS is
selected for you, so Track will run it. Your clicks on the
animal are not part of **Ctrl+Z**: take one back with *Remove this click* as
above. If an earlier Ctrl+Z step was waiting, the status line tells you that a
segment click has just replaced it.)

> **Can you skip this step?** Yes — outlining is optional, and nothing else
> waits for it. Skip it when you only want to follow clearly visible features
> like a painted mark or an eye, and **always skip it when the animal is only a
> few pixels across** (a small fish or a bird seen from far away, an insect in
> a wide shot): there is no silhouette to outline at that size, and the
> outlining software will latch onto something bigger nearby instead. Place a
> point on the animal with **N** and press **Track** — the tracker zooms into
> its own close-up around your points, so a few-pixel target still works. What
> you lose by skipping is only the silhouette landmarks (tail tip, midline,
> volume), which on smooth-bodied, well-resolved animals are most of the value.
> You can still group the points as an animal without a silhouette: press
> **＋ Animal** in LAYERS and give it points.

### Step 4 — Choose your landmarks

In the right panel, click **Skeleton ▾** (or open the **Skeleton** menu) and
pick the animal type closest to yours — the entries read *Use template: Lizard /
iguana (14 landmarks)* and so on. The landmark names appear in LAYERS under
the animal, and a notice over the video says which one to place first. The
skeleton goes on the animal selected in LAYERS (it asks when several or none
are selected; with no animal yet, one is made for it). Each landmark is named
after its animal ("iguana snout"; the tree shows just "snout"), so two animals
can share a template ("squirrel snout", "chipmunk snout").

Look at the small icons beside them:

- **A solid coloured square** — you need to place this one by clicking it.
- **A dotted outlined square** — this one comes from the silhouette. Do nothing;
  it appears on its own when you track — *if* you outlined the animal in Step 3.
  Without an outline it stays empty.
- **A coloured ring** — a ball marker (a ball the program outlines and follows
  as a circle; see section 10). No skeleton has one.
- **A greyed-out name** — not placed yet.
- **A ring** — a ball marker: the program outlines the ball and follows the
  centre of the fitted circle (see *Track the wand balls* in section 10).

The dotted, silhouette landmarks cannot be placed by hand: with one of them
selected, **N** and a click add a *new* point instead. To place such a landmark
yourself, right-click it → **Data source** → **Track by appearance (click to
seed it)** first.

Hover over a name and a line says how that landmark is found.

**Place the head landmark first.** It is usually called `snout` or `head`, and it
tells the program which end of the animal is the front. Without it, the program
has to guess, and on a legged animal it can guess wrong and swap head for tail.

To place a landmark:

1. **Click its name** in LAYERS to select it.
2. **Press N.** The cursor becomes a crosshair — the program is now armed for
   exactly one placement.
3. **Click the spot on the video.**

Repeat for each solid-square landmark you want. You do not have to place them
all — place the ones you need.

**Where does a new point go?** A point you add with **N** and a click (or with
**＋ Point**) joins an animal by itself: the animal selected in LAYERS; else
the animal whose silhouette is under your click on this frame; else the animal
of the point that is selected (so clicking point after point builds up one
animal); else **Scene**. When a silhouette decided it, or you clicked on
another animal's silhouette, a notice says where the point went — click the
notice to move it. You can always move points later by dragging them in LAYERS
(*Several animals* in section 7).

> **Why press N?** **N** is the way to *add* a point, and one press allows
> exactly one placement, so a stray click never creates one. For a landmark
> that is already in the list, a plain click without **N** does the same job:
> it puts the landmark selected in the list on the frame you are on (that is
> also how you correct it later — see section 7). With nothing selected, a
> plain click does nothing at all. If you change your mind after pressing
> **N**, press **Esc**.
>
> Do not select a name with a **dotted** square before pressing **N**: those
> come from the silhouette, and **N** would add a new, separate point instead.

### Step 5 — Track

**Select what you want tracked** in LAYERS — only that is tracked. Press
**Ctrl+A** to select everything (every animal and every point), or click rows
(Ctrl+click adds one, Shift+click a range). A point row is that point; an
**animal row is its silhouette and all its points**. The Track button says what
it will track, for example **Track · 3 points + silhouette ▶** or
**Track · 2 silhouettes ▶**. With nothing selected it tracks nothing and says
so. What is selected is part of your project: save, close and reopen it, and
the same rows are selected again.

A point selected on its own is tracked on its own: its animal's silhouette is
**not** tracked along with it. There are two exceptions, where the silhouette
comes along because the point cannot be followed without it: the animal has
**Keep its points on its silhouette** ticked (right-click the animal; off
unless you turn it on — section 14), or the point is a landmark that comes from
the silhouette (tail tip, midline …).

Each point is followed by **its own tracker**, shown beside its name (**AT**
AllTracker, **CT** CoTracker3, **MS** Moving spot). One Track press handles a
mix: Moving spot points, ball markers and silhouettes run together with the
AllTracker *or* CoTracker3 points; if both AllTracker and CoTracker3 points are
selected, they run **one after the other** over the same frames (the button
says *2 passes*), and a point that stops ends the run for all of them, so every
point of the run ends on the same frame. The pass that holds the head landmark
goes first and makes the silhouettes; the second pass reuses them instead of
outlining the animal a second time, so its on-body landmarks are kept on the
same outlines. A pass whose points have no position on the frame you are on is
left out, and the other still runs. One Ctrl+Z undoes it all.

If you start in the middle of a stretch that is already tracked, the program
asks before overwriting it, and counts **only the points being tracked** in
that question. A point you placed by hand on the frame you start from keeps its
white diamond. If a frame cannot be read once, the program tries it again; when
the second try works, the message says so and asks you to press **Track ▶**
again, instead of calling your file damaged.

Press the blue **Track ▶** button (or the **T** key). If it looks grey instead
of blue, nothing has a position on the frame you are on. Press it anyway and a
notice says what to do. Its **▾** menu (the run mode, Every camera, the point
model) still opens, so you can set those before anything is placed.

Now watch. The video plays forward frame by frame, the markers move with the
animal, the outline follows it (if you made one), and the timeline fills in from
left to right. The status bar along the very bottom shows the speed and an
estimated finish time.

**You can stop it at any moment:** press **X** or **Space**, or click the blue
button, which now reads **Pause ■ (X)**. Fix whatever looks wrong (the next
section shows how) and press **Track ▶** again: it carries on from the frame
you are on.

**It may stop on its own before the end.** That is the program telling you it has
lost something, or that a body part has left the animal's outline. It cuts that
body part's track at the first frame it stopped being trustworthy, jumps there
and says why — see the next section. It is doing its job.

---

## 7. Checking the result and fixing mistakes

**This is the most important section of the manual.** No tracker is perfect.
What makes results trustworthy is *your review*, and Kinetrace is built around
making that fast.

### The program stops by itself

With **Auto-pause** switched on (it is, by default), the program stops as soon
as it decides it has genuinely lost a body part. Deciding takes it a few
frames (a part that is merely hidden for a moment keeps its confidence; a lost
one does not, and the program waits until the two can be told apart), but the
result is always placed at the **first** unreliable frame:

- the lost body part's track is **cut** at that frame — nothing after it is
  kept, so what you see on the timeline ends exactly where it stopped being
  trustworthy,
- the playhead jumps to that frame,
- the body part is selected, ready for you to click it into place (it has no
  marker on that frame any more, so a plain click on the video puts it there),
- its row on the timeline ends at that frame; any frames just before it where
  the program was already unsure show **red**.

A brief hidden object does *not* trigger this — only a real loss.

With the **Moving spot** point model (Track ▾; see *Which point model should I
use?* in section 14) the stop comes on the very frame where the spot is not
where its speed put it, or where two spots look alike: it does not wait and it
never guesses. The track ends on the frame before. Click the spot where it is
and press **Track ▶** again; clicking it on the next frame too gives it its
speed. A spot that flies out of the picture simply ends, without a stop. When
a run holds other points too, they keep every frame they were tracked up to
the stop (they used to lose the last few). If your click was a little off a
tiny spot, Moving spot stops at once rather than following a glint beside it:
click it more precisely, or on a second frame.

While an outline is on, the program also watches for these, and stops or turns
the body part red rather than quietly exporting a wrong position:

- **The animal itself is gone.** With Auto-pause on, when there has been no
  outline — or the outlining model no longer believes the animal is there —
  for 16 frames in a row, the run stops and takes you to the first of them. If
  the animal is visible there, press **S**, click it, and press **Track ▶**
  again.
- **The outline jumped to another animal.** The outlining model, having lost
  your animal (it left the picture, or went behind something), sometimes picks
  up the next thing that looks like it — another bat of the same colony, say.
  When the outline comes back far from where your animal was heading, or jumps
  several body lengths in one frame, the run stops at the first frame your
  animal was missing, whether Auto-pause is on or off, and nothing of the other
  animal is kept. If your animal is visible on a later frame, go there, press
  **S**, click it, and press **Track ▶**. (A frame you clicked is always taken
  as your animal.)
- **A body part left the animal.** When the animal has *Keep its points on
  its silhouette* ticked (right-click the animal in LAYERS; see section 14), a
  part that leaves the silhouette stops the run at that very frame: the
  program will not pull it back onto some other part of the animal and carry
  on. Its track ends on the last frame it was on the body. A part you marked
  *May leave its silhouette (free point)* in its right-click menu is exempt.
- **Two body parts have merged.** With *Keep its points on its silhouette*
  ticked, two parts kept on the animal
  that were clearly apart when you clicked them and now sit on top of each
  other (say, eye and neck) both go red — skeleton landmarks and points you
  placed yourself with **N** alike. The program does not move them for you; a
  merge that lasts counts as a loss, so with Auto-pause on the run stops there.
  Go to the first red frame and click each one where it belongs.
- **The head landmark is not at the head any more.** The outline's midline is
  laid from the head landmark to the far end of the body. If that landmark has
  slid along the flank, the midline would start mid-body, so the program uses
  the body's own length instead, keeps head and tail consistent with the
  previous frame, and lowers the confidence of every outline-derived part on
  those frames to one half at most (hover the part's row on the timeline to see
  the number). Re-place the head landmark and track again.
- **A part at the edge of the picture.** When the outline touches the picture
  edge, a tail tip or foot found right at that edge is the edge, not the body
  part: it is left blank on those frames. Outline-derived feet and wing tips
  are never shown more than 70 % confident, and any outline-derived part that
  jumps more than 15 % of the body length between two frames drops to 30 %:
  that is what a swapped leg or a flipped tail looks like.

### What right-clicking a body part offers

Right-click its name in LAYERS, or **hold the right mouse button** on
its marker on the video for half a second (a short right click on the marker
clears it on that frame only — see below), and besides **Rename point** and
**Delete point** you get (hover an entry to see what it does). Wherever this
manual says *right-click a point*, either way works. (While the **Pan** tool,
**H**, is on, a right click on a marker does nothing at all: the tool is for
looking, never for editing.)

**Several points at once:** select them (Ctrl+click in LAYERS, **Ctrl+A** for
all) and right-click one of them in the list, or hold the right button on one of
their markers. The menu then acts on **all of them** — track only these; clear
them on this frame, in the selected frame window, or their whole tracks; delete
them; mark them hidden here; hide or show them; give them a tracker; fill their
gaps; **Move to** another animal — and when an animal's row is selected too,
its silhouette is cleared with them. With exactly two points of one animal
selected, **Connect them with a bone** draws a bone between them (again:
**Remove the bone between them**). Each is one Ctrl+Z step. A *short* right
click on one of their markers clears all the selected **points** on this frame
and keeps them selected; it never touches a silhouette, even when an animal's
row is selected (use the menu for that).

For one point:

- **go to its first or last frame**, and to its first or last hand-placed
  frame, with the frame numbers shown in the menu;
- **go to its first doubtful stretch** (the red part of its timeline row);
- **clear its position on this frame** — the one-frame delete, for when the
  tracker put a single frame in the wrong place;
- **clear its track inside the frame window** you selected on the timeline;
- **clear its whole track**, keeping the body part in the list so you can place
  it again;
- **Hidden on this frame (keep, do not export)** — see *When a body part is
  really hidden* below;
- **Fill its gaps between hand placements** and **Replace everything between
  its hand placements with that curve** — see *Keyframe digitizing* below;
- **Data source** — whether the part is followed by how it looks (**Track by
  appearance (click to seed it)**) or worked out from the outline (**From
  silhouette:** the tail tip, a point along the midline, the centroid, a foot
  or a wing tip — left and right as seen from above the animal, so they swap
  when it is filmed from below). An outline source needs the animal outlined
  with **S** and fills in when you track. Switching either way throws away the
  track the part has now, so if it has one the program first tells you how
  many frames that is (and how many of them you placed by hand) and asks;
  **Ctrl+Z** brings it back;
- **May leave its silhouette (free point)** — for a point of an animal that
  is allowed off its outline: *Keep its points on its silhouette* (section 14)
  then leaves it alone. (A mark that belongs to no animal at all, such as a
  reference on a rock, is better moved to **Scene**: Scene points are never
  held on a silhouette);
- **Move to ▸** — give the point to another animal, or to **Scene**. It is
  renamed "*animal part*" in every camera, as one Ctrl+Z step. A landmark that
  comes from the silhouette needs its animal, so it is not moved to Scene (a
  notice says why);
- **Use it as *animal*'s head** — the silhouette's midline is measured from
  the head, so this tells the program which end is the front;
- **Lock to seed appearance (re-anchor)** — off by default. The tracker keeps
  looking for the patch exactly as it looked where you placed it and snaps
  back to it when it finds it nearby: a cure for slow drift on a sharp, rigid
  mark. Plain points only, not regions;
- with a 3D calibration, **Snap to the other cameras' rays here** (section 10).

Each clearing entry, the curve fill, a change of data source, the snap and
**Delete point** is one **Ctrl+Z** step (renaming, *May leave its silhouette* and
the appearance lock are not undo steps: just switch them back). If the position is
right but the part was not really visible, use *Hidden on this frame*
(**Shift+X**) instead of clearing: that keeps the number but leaves it out of
your exported table.

### Several animals

You can track as many animals or objects as you like, each as its own
**animal** in LAYERS — two lizards on a rock, a bat and its prey, an animal and
the branch it lands on:

1. press **＋ Animal** in LAYERS: a new animal row appears, selected, with no
   silhouette yet (double-click it to name it);
2. to outline it, press **S** and click it on the video (outlining is optional:
   an animal can be just a group of points);
3. give it points: with its row selected, press **N** and click them on the
   video, or pick a skeleton with **Skeleton ▾**;
4. repeat for as many as you need.

Each animal has its own row, its own colour on the video, and its own lanes on
the timeline. **S** outlines the animal selected in LAYERS; with several
animals and none (or several) selected, the click does nothing and a notice
says to select one. A row's right-click menu works on that row. Select every
animal you want tracked (Ctrl+A selects all of them and every point) and press
**Track ▶**: all of them are tracked **in one pass**, each with its own
silhouette.

**Moving points between animals:** drag them in LAYERS onto another animal (or
onto one of its points), or onto **Scene**. They are renamed "*animal part*" in
every camera, as one Ctrl+Z step. The point's right-click menu → **Move to ▸**
does the same. A landmark that comes from the silhouette needs its animal: it
is not moved to Scene, and a notice says why.

Anything else that works on **one** animal — a skeleton template, a landmark
taken from the silhouette (*Data source*), *Import → Silhouettes*, *Carve
Volume*, the Body run's silhouette — uses the one selected animal; with several
selected (or none) it asks which. Exports of silhouettes write one file (or
folder of masks) per animal. A timeline clear of silhouettes that covered no
silhouette lane takes the selected animals, and asks before clearing every
animal's.

When one animal is lost, or the outlining model picks up something else in its
place, **that animal ends there and the others go on**; at the end of the run a
notice names it and the frame. Where it is visible again, select its row, press
**S**, click it, and Track.

Body parts belong to an animal, and each is named after it ("lizard snout",
"lizard 2 snout"), so two animals can have the same skeleton; its tail tip,
midline and feet are computed from that animal's silhouette, and with *Keep its
points on its silhouette* ticked its body parts are held on that silhouette. A
Shift+drag along one animal's silhouette lane on the timeline and **Delete**
clears that animal's silhouettes only. With several cameras, an animal made in
one camera appears (empty) in the others, to be clicked there; renaming or
removing it acts in every camera.

**Skeletons and bones belong to an animal too.** **Skeleton ▾** puts a template
on the selected animal. To draw a bone yourself, select two points of one
animal, right-click → **Connect them with a bone**; bones are drawn on the
video and in the 3D view. A point's menu → **Use it as *animal*'s head** sets
the head the silhouette's midline is measured from. When an animal's points and
bones are as you want them, right-click the animal → **Save its points and
bones as a skeleton template…**: another animal, in this video or a later one,
then gets the same points from **Skeleton ▾**. Deleting a point also removes
its bones.

### An animal in LAYERS

Each animal's row in **LAYERS** is the handle for everything about it — a
coloured swatch (filled once it has a silhouette, an outline until then; a dot
inside when its points are kept on its silhouette), its name, a checkbox, and
its points underneath:

- **the checkbox** shows or hides its silhouette on the video (the **Mask**
  button in the bottom bar hides every silhouette at once);
- **double-click the name** to rename it — its points are renamed with it, in
  every camera (the timeline rows follow);
- **right-click it** for the rest: *Rename the animal…*; *Outline it on the
  video (S)*; *Show its silhouette*; *Keep its points on its silhouette*;
  *Show the midlines*; *Jump to its first / last silhouette*; *Clear its
  silhouette on frame N*, *in the selected frame window*, or *Clear ALL of its
  silhouettes (keep the clicks)* (so pressing Track outlines the video again);
  *Save its points and bones as a skeleton template…*; *Forget its bones and
  head (keep the points)*; *Remove the animal (its points go to Scene)*; and
  *Remove the animal and its points*.

The line under the tree counts the selected animal's points, clicks and the
frames that have a silhouette, so you can see at a glance how far the
outlining got. **Ctrl+Z** undoes any of the clearing steps.

**Delete** (the button above the tree, or the **Delete** key when no frame
window is selected on the timeline) deletes what is selected. For an animal it
asks: *Keep its N point(s)?* — **Yes** moves them to Scene with their tracks,
**No** deletes them too. Removing an animal cannot be undone with Ctrl+Z (its
clicks and silhouettes go); deleting points can.

### Fixing a body part that has gone wrong

Go to the first frame where it looks wrong (the red stretch shows you where), and
then either:

- **select the point in LAYERS, then click** where it should be
  (Ctrl+click does the same), or
- **right-click its marker** to clear it on this frame only, when it is not
  really there (the right click also selects it).

To look around while zoomed in, **hold the left button and move** the mouse:
that pans the view, and never moves a point (starting on a marker too).

Each correction is one **Ctrl+Z** step. A body part that comes from the
silhouette (a dotted marker: tail tip, midline, feet) cannot be moved by hand:
the program leaves it where it is and says so in the bottom bar. Correct the
outline instead (press **S** on that frame and click the animal), or, to follow
that part by how it looks from now on, right-click it → **Data source** →
**Track by appearance (click to seed it)** (the program asks first, because that
throws away its silhouette track).

Then press **Track ▶** again. Tracking restarts from the frame you are on, using
your corrected position, and overwrites what follows. Your correction is treated
as certain truth. Only what is selected is tracked: while the part you
corrected is selected alone, the button reads **Track · 1 point ▶** and
re-tracks only that part. Press **Ctrl+A** first to re-track everything.

> **Only fix the *first* wrong frame.** Everything after it is being re-done
> anyway. Correcting frame after frame by hand is wasted effort.

### Swapping, moving, filling and splitting points

Sometimes the tracker follows the right body parts but under the wrong names:
the left and right feet swap where they cross, two balls on a wand trade
places, or a stretch of one landmark was tracked as another. **Edit → Point
Tools…** fixes which point is which, in the camera you are working in:

- **Swap two points.** A and B exchange their data over the frames you
  choose. Quickest way: select both in LAYERS (Ctrl+click), then
  right-click → *Swap these two points…*.
- **Move A's data to B.** On every frame of the range where A has data, B
  takes it (B's own data there is replaced) and A is cleared there: for a
  stretch that was tracked under the wrong name.
- **Fill B's empty frames from A.** Where B has no data and A has, B takes
  A's; A stays as it is. For one landmark tracked in two pieces: fill, then
  delete the spare point.
- **Split A into a new point.** From a chosen frame on, A's data becomes a new
  point named *A (2)* (double-click the name to rename it), and A ends on the
  frame before. Also on the point's right-click menu: *Split it into a new
  point from frame N*.

Choose the frames: the whole video, from the current frame to the end, the
window selected on the timeline (Shift+drag), or two frame numbers. Before you
press the button, the dialog says how many frames each point has data on in
that range. Each tool is one **Ctrl+Z** step.

Everything about a frame moves with it: the position, the hand-placed and
hidden marks, the confidence, and a ball's circle and clicks. Body parts that
come from the silhouette (dotted markers) are recomputed by every run, so they
cannot be swapped or moved: switch them to *Track by appearance* first. A ball
marker exchanges data only with another ball marker. Only this camera changes:
the same names in the other cameras keep their own data, so fix each camera
where its tracker swapped.

### Digitizing frames by hand

Sometimes the tracker simply cannot follow a body part through a stretch of
footage (a fast strike, a blur, a body part hidden behind a branch). For those
frames you can do what you would have done in a classic digitizing tool: place
the point by hand, frame by frame.

1. click the point's name in LAYERS,
2. go to the frame,
3. click where the body part is — the point moves there and the frame is
   recorded as **placed by hand** (a small white diamond appears on its timeline
   lane),
4. press **F** to go to the next frame and click again.

Every hand-placed frame is stored with full confidence and exported like any
other. **,** and **.** step between the frames you placed by hand, so you can
always return to where you left off, and right-clicking the point offers *go to
its first / last hand-placed frame* with the frame numbers. **Shift+<** and
**Shift+>** are the wider jump: the first and last frame the point has a
position on at all. **Ctrl+Z** takes the last click back.

A plain click only ever moves the point that is selected in the list. With
nothing selected it does nothing (a click on a marker just selects that point),
and holding the button and moving only pans the view, so an accidental click
cannot add points or edit a different one. Two quick clicks count as two clicks,
so you can digitize frame by frame as fast as you can press **F** and click. To add
a *new* point, press **N** first. A body part that comes from the silhouette
cannot be digitized by hand (see *Fixing a body part that has gone wrong*
above). Your click is kept to a fraction of a pixel, in the same pixel
coordinates the tracker uses, so a hand-placed frame lines up exactly with the
tracked frames around it (and every export converts both the same way).

Running **Track ▶** through hand-placed frames replaces them with the tracker's
output (the tracker restarts from the frame you are on and overwrites what
follows), so digitize by hand *after* the last tracking run over that stretch,
or track in pieces around it.

**Keyframe digitizing.** Some stretches are easier to do by hand than to
track: a few frames of blur, a limb behind the body. Place the part by hand on
a handful of frames spread over the stretch (its **keyframes**), then
right-click it → **Fill its gaps between hand placements**. A smooth curve
through your placements fills every frame between them that has no data. The
filled frames are stored at confidence 0.6 and get no white diamond (you did not
place them), so hovering the row on the timeline tells them from the frames you
placed. If the tracker did put something there but it was wrong, **Replace
everything between its hand placements with that curve** overwrites those frames
too (never your own placements). Both leave alone every frame you marked hidden
(**Shift+X**) — the bottom bar says how many; unmark them first if you want them
filled — and any frame where the curve would leave the picture. Both are one
Ctrl+Z step, and both respect a frame window selected on the timeline. The curve is a straight line with two keyframes and a
natural cubic spline with three or more — the more keyframes on a curved path,
the truer the fill.

### If a body part left the picture and came back

While it is outside the picture, its cells are deliberately left **empty** — not
guessed. That is correct: there is no position to record. Empty cells are what
you will see in your exported table, and your analysis software should treat them
as missing data.

When it comes back:

1. go to the frame where it reappears,
2. click its name in LAYERS,
3. press **N** and click where it now is,
4. press **Track ▶**.

Because that point currently has no position on this frame, your click
**continues the same body part** rather than creating a new one. The program
confirms this in the bottom bar. (If you wanted a genuinely new point instead,
press **Esc** before **N** to deselect.) A body part that comes from the
silhouette needs none of this: it comes back by itself when the outline does,
and **N** with it selected adds a new point rather than continuing it.

### Seeing what happened: trails, ghosts and the loupe

Three views help you judge a track without stepping through it frame by frame.
All of them are under the **View** menu and none of them changes any data.

- **Trails** (*View → Trails*) draw each body part's recent path as a fading
  line behind its marker. Choose *Off*, *Last 10 frames* (the standard) or
  *Custom…* and type how many frames you want (up to 1000). Tick *Also show the
  upcoming path* to see, dashed, where the point goes **after** this frame: a
  wrong step ahead shows before you get there. With several cameras the trails
  are drawn in the camera you work in; while **Track ▾ → Every camera** is
  ticked, every camera draws its own trails too — during the run and afterwards
  while you scrub to check it. *Off* shows no trail in any camera.
- **Onion skin** (**O**) draws a hollow ghost where every point was one frame
  ago (solid ring) and will be one frame on (dashed ring), joined to the marker
  by a dotted line. A point that jumped stands out immediately.
- **The loupe** (**L**) is a magnifier that follows your mouse over the video,
  with a crosshair on the exact pixel. Use it when placing a point at 4K: you
  can hit the right pixel without zooming the whole picture in. It follows the
  frame and the zoom you are on.
- **Show segment midline** and **Show skeleton bones** switch the line down
  the centre of the outline, and the bones joining an animal's landmarks, on
  or off. A bone is not drawn to a landmark that is hidden on that frame
  (Shift+X), so no line points at a place you said is not really visible.

### Making a faint animal visible

*View → Display filter* changes only what you **see**:

- **Enhance contrast** brings out a dim animal against water or foliage.
- **Brighten** lifts dark footage.
- **Frame difference** shows only what changed since the previous frame you
  looked at, so a small moving animal stands out from a still background. Step
  with **F** and **B** for a true one-frame difference.

The tracker always works on the original pixels, exports are unaffected, and
nothing is written to your video. Switch back to **None** when you are done.

### When a body part is really hidden

Sometimes a point has a position that is only a guess: the foot is behind the
branch, the tail is under the body. Rather than deleting it, mark it **hidden**:
select the point and press **Shift+X** (or right-click the marker → *Hidden on
this frame (keep, do not export)*). The position stays in the project, so you can
unmark it (**Shift+X** again), but it is left **blank in every export**, ignored
by 3D, and never written over by a curve fill (*Keyframe digitizing*, above).
Hidden frames show hatched on the timeline, the marker becomes a crossed ring and
its name reads *(hidden)*. Each mark is one **Ctrl+Z** step.

For a whole stretch: **Shift+drag** across the timeline (frames and rows, as for
clearing), then right-click the band → *Mark HIDDEN in frames …*. *Unmark hidden
in frames …* reverses it, and **Ctrl+Z** undoes either. If your drag touched no
body-part row, it applies to the parts selected in LAYERS — or, after
asking, to all of them.

### Regions: circle, rectangle, polygon

A region tracks a whole patch as one point (its fitted centre) — good for a
textureless body or a marker that is larger than a dot. The **▾** on the Point
button (**Point ▾**) chooses the shape:

- **circle** — press **N**, drag from the centre outward;
- **rectangle** — press **N**, drag a box;
- **polygon** — press **N**, click each corner in turn, then press **Enter**
  (or double-click the last corner). **Esc** abandons a half-drawn polygon.
  While *polygon* is chosen, a click after **N** is a corner, not a point:
  choose *circle* again before placing ordinary points (the choice is saved
  with the project).

Whatever the shape, keep the outline **on the object**: the region's inside is
what gets tracked, and background inside the outline pulls the centre off.

### Undo

**Ctrl+Z** (*Edit → Undo Last Run / Edit*) takes back the last change you made
to the data, and every correction counts as one step of its own: a tracking run
(or one semi-automatic step), a hand placement — a click, a Ctrl+click,
or an **N** click that continued a point — a **Shift+X** mark, clearing or
hiding a stretch (body parts or silhouettes), deleting points, a change of a
point's data source, a curve fill, or a snap onto the other cameras' rays. Your
clicks with the segment tool (**S**) are the exception: **Ctrl+Z** does not take
them back (right-click a click → *Remove this click* does), and the status line
says so.

> ⚠ **Undo is only one step deep.** There is no second undo, and removing an
> animal cannot be undone at all (the program asks first). Before anything
> drastic, save (**Ctrl+S**).

### Going one frame at a time

For a really difficult stretch, use **Semi-automatic** mode: click the small
arrow **▾** on the Track button and choose *Semi-automatic*. The button then
reads **Step ▶ (F)**, and each press of **F** (or **T**) advances exactly **one**
frame and stops. Check it, fix anything, press **F** again; each step is its own
**Ctrl+Z**. Slow, but nothing escapes you.

### Clearing a bad stretch

If a stretch is beyond fixing, delete it rather than leaving bad numbers in:

1. **Hold Shift and drag across the timeline** — drag sideways across the frames
   you want gone, and up or down across the *rows* you want affected.
2. Press **Delete**.

What gets deleted depends on which rows you dragged over: the silhouette row
clears the outlines, a body-part row clears that body part's positions, dragging
across both clears both. The highlighted band tells you in words what Delete will
remove before you press it. Right-click inside the band for the same choices
spelled out, plus *Mark HIDDEN* / *Unmark hidden* for the window and *Extend
selection to every lane*. **Ctrl+Z** brings it back. A drag along the frame numbers at the
top that touches no row clears the body parts selected in LAYERS — or,
after asking, all of them.

---

## 8. Reading the timeline

The timeline is the panel between the video and the bottom bar. It is both your
map of the work and your scrubber.

**Each horizontal row is one body part** (plus a strip along the top for events
and notes). The rows follow LAYERS: each animal's silhouette row (if it has a
silhouette), then its points' rows, indented and named by part, then the Scene
points' rows. Reading along a row from left to right is reading through the
video. Click the name at the left end
of a row to select that body part; the selected part's row is drawn brighter.

- **A solid coloured bar**, in the body part's own colour — this body part has a
  position on those frames (the bar is faint while its tick in LAYERS
  is off).
- **A gap** — no data there. Either not tracked yet, or deliberately empty
  because the part was out of the picture.
- **A thin line along the middle of a row** (several cameras only): this camera
  has no data there, but **another camera** has this body part at that instant.
  The rows are always the **working camera's** — the one outlined in blue, bold
  in CAMERAS. Hover the line to see which cameras have it; click that camera's
  view to work in it, and its track becomes the solid bar.
- **A red tint** — tracked, but the program was *unsure*. **Look at these
  stretches.** Red is not necessarily wrong, but it is where errors live. On
  the silhouette row, red means the outlining model was unsure the animal was
  there.
- **The white vertical line** is the frame you are looking at.
- **A small white diamond** on a row is a frame you placed by hand.
- **A hatched patch** is a stretch you marked as hidden (kept, not exported).
- **A small green triangle** in the strip along the top is a note; hover it to
  read it, right-click it to edit it, jump to it or delete it.
- **A magenta strip along the bottom of a row** appears only after a 3D
  reconstruction (section 10): on those frames this camera's position for the
  body part disagrees with the other cameras' by more than about 5 pixels (more
  on large pictures), so it has probably slid onto another spot. Hover it: the
  pop-up gives the disagreement and the limit in pixels and names the two fixes —
  right-click the part on the video → **Snap to the other cameras' rays here**,
  or **3D → Re-track Disagreeing Stretches…**.

**To move along a body part's track:** with it selected, **Shift+<** and
**Shift+>** go to the **first and last frame it has a position on**. With
nothing selected they go to the first and last frame of the silhouette instead
(with several animals: of the animals selected in LAYERS, else of all).
**,** and **.** step between the frames you placed by hand. **J** jumps to the
next red (low-confidence) stretch and **Shift+J** to the previous one — of the selected points, or of every point when nothing is
selected. That is the fastest way to review a long video: press **J**, look, fix
or mark hidden, press **J** again.

**To move through the video,** click or drag anywhere along it.

**To see individual frames,** zoom the time axis: hold **Shift** and press
**+** or **−** (or hold **Ctrl** and scroll the wheel over the panel). Zoomed in
far enough, every frame gets its own column of pixels and you can click precisely
the one you want. A small scroll bar appears along the bottom edge showing where
you are in the whole video — drag it to move, or drag the timeline itself with
the middle mouse button. The three magnifier buttons in the timeline's top-left
corner, left of the frame numbers, zoom it out and in; the last one (**⤢**)
shows the whole video again.

**To make it taller** (more rows visible at once), drag the divider between the
video and the timeline upward. The mouse wheel scrolls any rows that still do not
fit.

**Hover over anything** for a plain-language explanation of what is under the
cursor. Over a body-part row it gives the frame, the time, the confidence,
whether you placed it by hand or marked it hidden, and — after a 3D
reconstruction — how well this camera agrees with the others.

---

## 9. Marking events

Events are named stretches of frames — "dive", "takeoff", "stride 3". They are
for *your* navigation and analysis; they do not affect tracking at all.

To mark one: press **E** at the start frame, move to the end frame, press **E**
again, and give it a name. A coloured ribbon appears along the top of the
timeline.

Marking the same thing again later? The naming box offers the names you have
already used. Pick one and it becomes another occurrence of that event, sharing
its colour. The **Events** menu groups all occurrences of each name together.

Click a ribbon to jump to that event. **Right-click it** (also right after you
have clicked it) to jump to its start or end, select its frames on the timeline (to clear or mark hidden), rename it, add
a note to it, move its start or end to the frame you are on (*Set start to
current frame*, *Set end to current frame*), or delete it. Events and notes are
not part of **Ctrl+Z**: a deleted event stays deleted. Events are saved with
your project and are included in your exported results as a separate file.

---

### Notes and who did what

**Shift+N** attaches a free-text note to the current frame ("tail hidden behind
rock", "second animal enters"). Notes appear as small green triangles at the top
of the timeline, in the **Events** menu, and in the events sidecar file of every
export. Right-click an event ribbon → *Edit note…* to add a note to an event.

Tell the program who you are once — **Edit → Annotator Name…** (also in
Settings) — and every event and note you add records your name, so a lab with
several people digitizing can see who marked what. The name is remembered on
this computer for the next project, and applies to every camera of the project
(not just the one you are working in).

---

## 10. Filming with several cameras

Skip this section if you use one camera.

One camera gives you positions on a flat picture. Two or more cameras, filming
the same animal from different angles, let you calculate true **3D** positions:
right here, with the **3D** menu (calibrate the cameras, then reconstruct — see
*From several cameras to 3D* below), or in DLTdv, Argus or your own code, for
which Kinetrace exports everything they need.

### The problem: cameras do not start together

Unless your cameras were electronically synchronised, they each started recording
at a slightly different moment. So frame 100 of camera A and frame 100 of camera
B are **not the same instant**. Before the views mean anything together, you have
to establish how far apart they are.

That difference is a single number per camera, called the **offset** — a
whole number of frames when you line the cameras up yourself; the program can
add a fraction of a frame later (see the note at the end of *Setting it up*).

### The first camera is the reference

Offsets have to be measured *against* something, and that something is **the
first camera you load**. It is marked **(reference)** in the panel, its offset is
**always 0**, and you cannot change it — there is nothing to change, it *is* the
zero.

(For the same reason the reference camera's offset box and its **◂ ▸**
nudge buttons are greyed out.)

Every other camera's offset then reads as one plain sentence:

> **"when camera 1 is at its frame 0, this camera is at its frame N."**

So a camera switched on **later** than camera 1 has a **negative** offset. If
camera 2 started 50 frames after camera 1, it reaches its own frame 0 only when
camera 1 is already at frame 50 — at camera 1's frame 0 it was, so to speak, at
frame −50 — so its offset is **−50**. A camera switched on **earlier** has a
positive offset: +50 means it is already at its frame 50 when camera 1 starts.
Hover the offset box for the same reminder. That meaning never changes,
including when you switch which camera you are working in — the numbers stay
put.

If you want to shift the whole set in time, change the *other* cameras. If you
remove the reference camera, the next one takes over as the zero and the
remaining offsets are renumbered against it — nothing actually moves in time,
the numbers just get measured from the new starting point.

### Setting it up

1. **All the cameras' videos in one folder?** Use **File → Open Folder of
   Videos…** and pick the folder. Every video in it is listed with its picture
   size, frame rate and length. Tick the ones that belong to this event (*Include
   subfolders* looks deeper), and put them in **camera order**: select a
   video and press **Move up** / **Move down** (or Alt+Up / Alt+Down); **Sort
   by name** goes back to the folder's order. The **Camera** column numbers the
   ticked videos: **camera 1 is the reference** that every other camera's offset
   is measured against, and the order is the order of the cameras in
   calibrations, 3D and every export — keep it the same as your calibration's
   (a DLTdv, easyWand or Argus file lists its cameras in order). Leave *Save the
   project now as* ticked and the project is saved at once as a file named after
   the folder, inside it — the file remembers which videos you chose and their
   order. If a project of that name is
   already there, the program asks before it saves over it (**No** leaves your
   cameras open and unsaved, and *Save Project As…* lets you choose another
   place). Each camera is named after its file; when files share a name (a
   `GX010001.MP4` in each of eight camera folders), the cameras are named after
   their folders instead (or get a *(2)* after the name), so every camera has
   its own name — which matters, because camera offsets are matched by name.
   Then go on at step 3.
   Otherwise: open the first camera's video normally. **This one becomes the
   reference**, so if it matters to you which camera the numbers are measured
   against, load that one first.
2. In the **CAMERAS** panel at the top of the right panel, click
   **＋ Add video** and choose the others. You can add several at once, up to 15
   cameras in total. A short list then shows the project's cameras and the new
   ones (marked *new*, in name order — not the file window's order): put them in
   their **camera order** (Move up / Move down, or drag a row) and press OK.
   Each appears as its own view in a grid above the timeline. Got the order
   wrong? **Camera order…** under the CAMERAS list changes it at any time: each
   camera keeps its points, offset, frame rate, lens and calibration (if another
   camera becomes camera 1 the offsets are re-measured from it — the cameras
   stay in sync); only a 3D result is dropped (Reconstruct again). With a
   calibration of the first cameras only, those must stay first.
3. **Line them up.** The quick way: **3D → Sync Cameras (Sound / Motion)…**.
   It offers two methods and a verdict for each camera; both need no tracking
   and no calibration.

   **Sound** is the usual way, and the one to try first. Every camera's
   microphone heard the same claps, voices and knocks, so their sound tracks
   can be slid against each other until they match. It works even for cameras
   that see nothing in common. Recordings are noisy — fans, wind, traffic —
   so there is a **filter**: only sounds between two frequencies are used
   (300 to 3800 Hz by default, where claps and voices live; lower the top to
   cut hiss, raise the bottom to cut rumble), and the *noise-robust* box
   listens to the *timing* of sounds rather than their loudness, so a loud
   fan cannot drown a clap. Keep it ticked unless a quiet room gives WEAK.
   **Stand on a clap first** (or any loud moment): the stretch it listens to
   is centred on the instant on screen when you open the dialog, whichever
   camera you are working in — the box *Around frame of …* shows that instant
   as a frame of camera 1. 40 seconds is the default length; a stretch with
   claps beats the whole recording, because minutes of pure room noise dilute
   them. Press *Find the offsets*. One thing to know: sound takes time to travel — about
   1 frame per 1.4 m at 240 fps — so a camera several metres farther from
   the clap hears it a few frames late. The dialog says so; the motion method
   and, after tracking, *Estimate Sub-frame Offsets* have no such bias.

   **Motion** reads a stretch of every video at postage-stamp size and compares
   *how much the picture changes* from frame to frame in each camera — a wand
   swung into view, a person walking past, a flash all leave the same bump in
   every camera that saw them. It needs a stretch in which something moved
   that all cameras could see; it is the choice when a video has no sound
   track, and a good cross-check of the sound result. Reading a camera's
   stretch takes seconds from a local disk and a minute or two over a network
   share. It looks in exactly the window you ask for: *Around frame of …* and
   the length you set, no more.

   Either way, GoPro-style file names carry the recording clock to the
   second, and the dialog uses it: it tells you what the clocks say and
   searches a couple of seconds around that. Each camera gets a verdict:
   **CLEAR** (use it), **WEAK** (check by eye first) or **NONE** (nothing all
   cameras heard or saw in that stretch — pick a stretch around a clap, a
   flash or the wand entering, tick *Use the whole recording instead*, widen
   the search, or switch method; a NONE row starts unticked). The line under
   the button also names a video with no sound track (use Motion for it), a
   reference camera whose stretch is silent, a file that could not be read at
   all (hover its verdict for the reason: it is not the same as having no
   sound), and a camera that did not record that stretch at all (choose a
   stretch every camera recorded). **Cancel** stops a running search at once,
   also while the sound is being read. Press *Apply
   the ticked offsets*. The hand check afterwards
   is the same as the manual way: step to a moment every camera saw and
   confirm it appears at the same time in each view.
   The manual way, for a stubborn camera: move to a moment that *every* camera
   recorded — a flash, a hand clap, the animal first touching the ground,
   anything unmistakable. Then for each other camera, tap **◂** or **▸** to
   shift it a frame at a time until it shows that same moment. If it is
   already showing the right moment, just press **Align here** and the offset
   is worked out for you. Align here works on the reference camera's row too:
   the reference never moves (it is the clock), so it sets the offset of the
   camera you are working in, and every other camera keeps its own.
4. The panel then reports the **overlap** — the stretch of frames for which every
   camera has a picture. 3D needs only two cameras to see a landmark at the
   same instant, so it can reach a little beyond that stretch.

> Cameras may record at *different frame rates* and *different resolutions*.
> A camera at twice the reference rate shows **×2** next to its offset: it steps
> two of its own frames for every frame of the reference, and its offset is
> counted in its own frames. Nothing else changes for you. Each camera's rate is
> the **fps button** in its row: if a file gives the wrong rate, click it and
> enter the real one (see [section 2](#2-what-you-need-before-you-start)); the
> ×n and every time in seconds follow, and the offsets stay as they are.

> Offsets you set by eye are whole frames. Cameras that were not electronically
> synchronised actually differ by a *fraction* of a frame as well, and for 3D
> that fraction matters (a fast animal moves millimetres in a hundredth of a
> frame). Section 10's **"From several cameras to 3D"** below shows how the
> program measures it for you; you never type a decimal yourself.

### Working with several cameras

**You track one camera at a time.** The camera you are working on is the one
highlighted in the panel — click any view to switch to it. Everything else on
screen — the LAYERS list, the silhouette, the timeline, undo — always belongs to
that camera. The other views follow along, showing the same instant, with their
own markers drawn on them, but you cannot edit them until you switch.

Switching cameras keeps you on the same instant, so you never lose your place.
A plain click on another camera's picture **only switches** to it — it never
places anything, so looking at a camera can never put a point in it. With **N**
(the **Point** tool) armed it is different: you have said you want to place a point, so the
click switches to that camera **and places the point there**, in one go. The
segment tool (**S**) and an event whose start you had marked with **E** are put
down by a switch, so a click meant for one camera's silhouette can never land in
another. The toggles in the bottom bar (Auto-pause, ROI, Follow, Mask, the
point model, semi-automatic mode) and the *View* settings — marker size, trails,
the display filter and the like — stay as you set them: they belong to you, not
to a camera. The selected body part comes along too: select *snout* in one
camera, click another camera, and *snout* is selected there, ready to be
clicked. The one thing a switch forgets is the undo step — **Ctrl+Z** only
reaches back to what you did since you last switched.

Each camera's row in the CAMERAS panel is one line — its number, name and
offset — so all of them fit; **▶** opens it: **Align here** and **×** on the
first line, and the offset box with **◂ ▸** (and ×2 for a camera recording at
twice the reference rate) underneath. **×** removes that camera from the
project; it asks first when the camera has tracked frames, and its tracks go
with it.

**Every camera has the same list of body parts, in the same order.** A point you
add in one camera appears in LAYERS in every other camera straight
away, under the same animal, greyed until you place it there: select it in that
camera, click it on the video, and track it. **＋ Point** (above the LAYERS
tree) makes a point with no position yet, already selected, in every camera —
the way to define your body parts first and then click each one in each camera.
Renaming, deleting or moving a point to another animal does the same in every
camera (the delete question says which other cameras lose their tracks, and
**Ctrl+Z** brings them all back). Animals are shared by name too: an animal
made in one camera appears (with no silhouette) in the others. A skeleton chosen
in one camera is given to the others, and a camera added later receives the
whole list.
That shared list is what lets the cameras be combined in 3D: the program matches
them by name.

**How the other cameras follow the playhead** — *View → Other cameras*, or the
**Sync all** button in the CAMERAS panel:

- **Sync all views** (the usual choice): every camera shows the instant the
  playhead is on, with its markers and guides. When you step one frame (**F**,
  **B**, a click on the timeline) every camera's new picture appears at the same
  moment, so what you compare is always one instant.
- **Active view only** (**Ctrl+Shift+2**, or untick **Sync all**): only the
  camera you are working on reads its video — much faster with many 4K cameras.
  The others stay on the picture they last showed, dimmed, with "not following
  (Active view only)" and that picture's frame number in their caption, however
  you move — playing, scrubbing or a single step. Tick **Sync all** again (or
  press Ctrl+Shift+2 again) and they come back to the playhead; click one of
  them to work in it. The dashed lines and ◇ in your working camera keep
  working, because they come from the other cameras' tracks, not their pictures.
- **Only the working camera** (**Ctrl+2**): the others are hidden and give the
  whole window to the camera you are working on.

**Which cameras are on screen, and where.** With many cameras every view is
small; show only the ones you need — two or three to check that a flash lines
up, say — and they get the room:

- the **eye** on a camera's line in CAMERAS shows / hides its view. A hidden
  camera is not read from its video (scrubbing is faster), but its tracks still
  count for 3D, the dashed lines and the ◇. The camera you are working in is
  always shown: clicking a hidden camera's line works in it, and it hides again
  when you go back to another camera (its eye stays off). With **Track ▾ → Every
  camera**, hidden cameras are tracked too, without being drawn — Track and a
  warning say so.
  **Show all (n hidden)** under the list (or *View → Other cameras → Show every
  camera*) brings them all back.
- **drag a view by its title bar** onto another view to put it there; the views
  between shift along, as in a video call. This only moves the pictures on
  screen — the cameras keep their numbers. Right-click a view's title bar to
  hide it, show every camera, or **arrange the views in camera order** again
  (also in *View → Other cameras*).

Which views are shown and where is kept with the project (it never counts as an
unsaved change).

**Tracking the same points in every camera.** When a body part (or a ball
marker) is placed in several cameras, you do not have to track it camera by
camera: tick **Track ▾ → Every camera** and the Track button says how many
cameras a run will cover ("Track ▶ · 3 cams"). Press Track (**T**) and the
selected points (**Ctrl+A** selects all of them) are tracked in each camera that
has them at this instant, **all cameras at the same time**, each one moving on live
in its own view; a camera where a point was not placed is simply skipped for
that point (with *View → Trails* on, each camera draws its trail as it goes).
**A hidden camera is tracked too** — one whose eye is off in CAMERAS, or every
other camera with *Only the working camera* on — but it is not drawn, so you
cannot watch the tracking there or stop it on a mistake you see. The Track button
says so before you press it ("· 5 cams (2 hidden)"), and a warning names the
hidden cameras when the run starts; click the warning to show them all.
The Track ▾ menu stays open while you tick, so the run mode
(Automatic / Semi-automatic), **Every camera** and the point model (AllTracker /
CoTracker3) can all be chosen in one visit; they combine. If only the camera you
are working in has the point at this instant, Track says which camera lacks it
(and on which frame) and tracks this camera alone. In semi-automatic mode **F**
steps one frame in every camera. **Shift+T** does one such run without ticking
anything. **X** stops every camera at once. One **Ctrl+Z** undoes the whole run
in every camera. If a point is lost in one of the cameras, the others are still
tracked, and at the end the playhead goes to the camera and frame where it was
lost, as after any auto-pause. (The cameras take turns on the graphics card a
frame at a time, so a run needs the memory of one camera, not of all of them,
and each camera's result is exactly what tracking it alone gives.)

### Calibrating the cameras yourself: the wand

Before any 3D can be computed the program must know **where each camera is,
which way it looks, and how strongly its lens magnifies**. Working that out is
called *calibration*, and you can do it entirely inside this program with a
**wand**. You do not need to understand the mathematics; you need a wand, a
short recording, and ten minutes.

**Build a wand.** Take a stiff rod and fix a clearly visible marker at each end
— two table-tennis balls painted a colour that stands out, or two bright dots
of tape on a dark rod. Measure the distance between the two markers, **centre
to centre**, as carefully as you can with a tape measure. 40 to 60 cm is a good
length for a volume about the size of a table; make it longer for a larger
volume. That measured length is the one number the whole calibration hangs on:
if it is 2 % wrong, every 3D distance you ever measure will be 2 % wrong.

**Film it.** With **all cameras recording at once** — the same set-up, the same
positions, the same lenses and zoom you will use for the animal — walk the wand
slowly through the whole space where the animal will be. Tilt it every which
way: horizontal, vertical, diagonal, towards one camera then another. Spend at
least half a minute; cover the corners of the volume, not only the middle. Then,
in the same recording, **drop a small object** (a ball) somewhere in view of all
cameras and let it fall for a good half metre: that tells the program which way
is *up* and gives it an independent check of the scale. (If you cannot drop
anything, put three marks on the floor instead: one for the origin, one along
the direction you want to call +X, one towards +Y.)

**Track the wand balls with SAM.** The two markers must be tracked in every
camera under the *same two names* (say "wand A" and "wand B"). Do not track a
ball as an ordinary point: a uniform ball gives the appearance tracker nothing
to hold. Use **Point ▾ → Ball marker** instead and click the middle of each
ball. On every frame the program outlines the ball with SAM (the same model
that outlines the animal), fits a circle to that outline and takes the
circle's **centre** as the point — exact to a fraction of a pixel even when a
hand or the rod covers part of the ball. Add one ball marker per ball; they
are tracked together in one pass, inside one 960-pixel window that follows
them. Balls too far apart for one window (more than about 740 px, such as a
long wand close to a 4K camera) are split into groups, each with a window of
its own, and brought back together when they come close again. Nothing to
do on your side. Each extra window makes tracking a little slower. Balls as
small as about 5 px across work. The fitted circle is drawn on the
video so you can see what was found. If a ball leaves the picture its track
simply ends and resumes when you click it again where it reappears (select
it, Point ▾ → Ball marker, click, Track); if the program loses a ball that is
still in the picture it stops there and tells you (with *Auto-pause* off it
carries on, and names each such ball and the frame its track ended once the
run is over). A third ball on the wand
(or a dropped ball) is tracked the same way and can serve as the *extra
point* or the *dropped object* in the calibration.

**One camera only?** The wizard still opens: its first two pages list the
landmarks and count the frames with both wand ends, so you can check your
tracking. It cannot calibrate from one camera — the two ends only fix a
direction and a scale when a second camera sees them too — and says so
instead of going on. Add the other cameras' videos when they are tracked, or
export this camera's tracks (Ctrl+E) for easyWand / DLTdv.

**Does the lens itself need calibrating?** Every lens bends the picture a
little; wide-angle lenses and action cameras (GoPro and the like) bend it a
lot — straight lines curve near the edges of the frame. The wand works out
where the cameras are and how strongly they magnify, but it assumes straight
lines stay straight. If your lens bends them by more than a few pixels, anything
tracked near the edges would land in the wrong place in 3D by about that much.

- **Ordinary lens** (a camcorder, a phone at 1×, a DSLR with a normal lens): not
  needed. The wand alone is fine.
- **Wide-angle or action camera** (GoPro, "wide" or "superwide" modes, any
  fisheye): needed. Do it once per camera and zoom setting; the profile is
  reused afterwards.
- **Not sure?** Do it once; the report tells you in pixels how much your lens
  bends the edges, and whether it matters.

**3D → Calibrate a Lens (checkerboard)…** walks you through it, with or
without a video open. Best is to have the project with your cameras open, so
the finished profile can be attached to one of them straight away (the wand
wizard's *The cameras* page also has a **Calibrate…** button per camera that
opens the same wizard). With nothing open, the wizard still runs — print the
board, pick the board video, check the boards — and its last page tells you to
**save the lens file**, because there is no camera to attach it to yet; if you
close it without saving, the program offers to save it for you. The wizard:

1. Press *Save the checkerboard to print…* (any folder works, also one with
   accents in its name) and print the file at **100 % /
   "actual size"**. Glue it to something perfectly flat and stiff — foam
   board, a clipboard, a piece of glass. Measure one square with a ruler (it
   should be 24 mm) and type what you measure. If you use a board of your own,
   keep **one count of squares odd and the other even** (the printed one is
   10 × 7 squares, 9 × 6 inner corners). That is what lets the program tell
   which way up the board is: turned 180 degrees, such a board swaps its
   black and white squares, while an 8 × 6 board looks exactly the same and
   nothing can say which corner is which. The lens does not care, but the
   drawn axes will flip and the board cannot be used to line cameras up.
2. With the camera set exactly as for the experiment (same zoom, resolution
   and frame rate), film the board for 20–30 seconds: fill about a quarter of
   the picture with it, move it slowly to **every edge and every corner**, near
   and far, and **tilt it** left, right, up and down by 20–40 degrees. Keep it
   sharp. **Hold the board by its edge or by a handle behind it**: fingers
   over the squares, or over the white border around them, make that frame
   unusable, and the whole board — border included — should stay inside the
   picture. Turning the board is fine: the program recognises its corners by
   the black square beside them, so the axes stay attached to the board
   however it is held. Holding the board still does not help: what counts is
   how many *different* poses it was shown in, not how many frames.
3. On the page *The checkerboard video*, press **Choose…** and pick that video.
   (The box starts with a video that is already open — usually the wand or
   animal recording, which has no board in it; replace it unless it is the
   board video.) Under *This is the lens of camera* pick the camera, check
   *Board size* (inner corners — 9 × 6 for the printed board), type the square
   you measured under *One square*, and leave *Lens type* on *Not sure* unless
   you know. Press **Find the boards**: it finds the board on its own (the
   board size cannot be changed while it is looking). The
   board video must have **exactly the same picture size** as that camera's
   video (same resolution and recording mode); if it does not, the page names
   both sizes and will not go on — film the board in the experiment's mode, or
   pick the camera it belongs to.
4. **Check the boards.** The next page shows you every board it found, so you
   can see for yourself what the calibration is being built from — see below.
5. The last page shows the verdict with its reasons, a before/after picture,
   and a map of where the board went (the boards the fit used). Press
   **Attach to** *camera* and the profile belongs to that camera of the open
   project (with nothing open that button reads **Close**); **Save lens
   file…** keeps it for other projects (`.klens.json`) —
   worth doing every time, since it stays valid as long as the camera keeps
   the same lens, zoom and resolution. Forgot? **3D → Export Lens Profile…**
   saves any camera's profile at any time (however it was made: checkerboard,
   GoPro's lens model, a file), as a Kinetrace `.klens.json` or for other
   programs (OpenCV `.yml` / `.json`, Argus `.txt` for a standard lens), and
   **3D → Load a Lens Profile for This Camera…** puts a saved profile on the
   camera you are working on (same picture size only; replacing a profile is
   asked first). Already have a profile? On step 3 press
   **I already have a lens file…** instead (a `.klens.json`, or an Argus /
   DLTdv camera profile `.txt`): it goes straight to this last page. From an
   Argus file with several lines each camera gets the line with its own camera
   number, and a camera the file has no line for is refused.
   **Several identical cameras?** Cameras of the same model, with the same lens,
   zoom and recording mode, can share one checkerboard calibration. When other
   cameras of the project record the same picture size and have no lens
   profile yet, this page offers **Also use it for the other cameras with …
   pictures** (it names them). Tick it and the button reads *Attach to cam1 and
   9 more*. Leave it off for cameras with other lenses or zoom settings: the
   same picture size does not prove the same lens.

**Two hand checks on the last page.** The report says how many pixels the
lens curves the picture edges by: how far an edge pixel sits from where a
straight-line lens would put it. That is the lens's own curvature, not an
error (the *fit error* is what measures the calibration): a GoPro-style lens
gives several hundred to about a thousand at 2.7K, a phone or camcorder a few
tens. It also says what **field of view** the model
implies corner to corner; compare that with the camera's own figure (a GoPro
"Wide" is roughly 120–150 degrees, "Linear" about 90–100, a phone 70–85). If
instead it says the model **runs away in the corners**, the curve is being
stretched past where the board ever went: the board reached too little of the
picture, so out in the corners the model has nothing to hold on to and would
place a point at infinity. That happens most with the fisheye model. Film the
board again pushed into every corner, or use the standard model if the report
says it fits about as well. The wizard never picks a runaway model on its own: with *Lens type* on *Not
sure* it fits both models to the very same boards, and when it leaves the
fisheye out because it runs away at the edges, the report says so. The pixel
limits in the verdicts are scaled by the picture's longer side, so a portrait
video is judged like a landscape one.

---

### GoPro footage

Kinetrace recognises a video straight from a GoPro (HERO, Mission) by the
information the camera writes into the file, and only then offers what follows.
Videos from other cameras are not read this way. A copy of a GoPro video that
another program exported or trimmed usually loses the settings and the lens
model: work from the files as they came off the card.

When a GoPro video opens, the program reads its recording settings (model,
resolution, frame rate, lens mode such as Wide or Linear, stabilisation, shutter
speed and ISO), how the camera was tilted (its gravity sensor), whether it was
knocked or moved during the recording, whether it dropped frames, its start time
(timecode), and GoPro's own model of the lens for that recording mode.

It warns you, once, about what can spoil 3D:

- **Stabilisation was on** (HyperSmooth or EIS): the camera re-warps every frame,
  so no lens calibration holds. Record with stabilisation off.
- **Dropped frames:** frame numbers after them no longer line up with the
  other cameras.
- **The camera moved during the recording:** a calibration made before the move
  does not hold for that camera after it. The notice gives the frame.
- **Settings that differ between cameras:** lens mode, frame rate or shutter
  speed (a fast animal blurs differently in each camera).

Click the notice, or use **3D → GoPro Cameras…**, for a table with every
camera: settings, tilt, dropped frames and when it moved (click a *Moved* cell
to go to that camera and frame).

**GoPro's lens model.** Every GoPro video carries the maker's own model of how
its lens bends the picture, for the exact mode it was recorded in, over the
whole picture including the corners. Kinetrace uses it in two ways:

- **Without a checkerboard:** 3D → GoPro Cameras… → *Use GoPro's lens model for
  the cameras without a lens profile*, or *Use GoPro's lens without boards* in
  3D → Calibrate a Lens. This is the lens *design*: each camera differs from it
  by about 1 % in focal length and up to about 10 px in where the centre of the
  lens is. The wand calibration fine-tunes the focal length.
- **With a checkerboard (recommended):** in 3D → Calibrate a Lens, choose
  *GoPro lens from the video + your boards* (chosen for you on GoPro footage).
  GoPro's curve is kept and only this camera's focal length and centre are
  measured from your boards, so the board does **not** need to reach the
  corners of the picture, the hardest part of filming it. When the boards
  cannot pin the centre down (two halves of them disagree), the program keeps
  GoPro's and says so: tilting the board more, up and down as well as left and
  right, measures it.

**Syncing GoPros.** When every camera has timecode, 3D → Sync Cameras starts its
search from the cameras' start times instead of the file names. The camera
clocks are set only to about a second, so this is where the search starts, not
the answer.

### Checking the boards before you trust the lens

A lens calibration is a handful of numbers, and no one can tell by looking at
them whether they are right. So the wizard shows you its evidence: **every
board it found, in a list you can scroll through.**

On each picture:

* **green dots** — the corners it found;
* **a light-blue line along the first row, and an orange ring around corner 0** — the order
  it found them in. Corner 0 is always the inner corner **beside the black
  square**, so on a board with one odd and one even count it is the same
  physical corner on every board, however the board was turned — the line
  above the pictures says whether that held for every board found. If one
  board is ringed at a different corner, it was read wrongly and its numbers
  will fight the others: untick it. (With a board that looks the same upside
  down, the ring simply sits at the corner nearest the top-left of the picture,
  and the line says so.);
* **red, green and blue arrows** — the board's own X, Y and Z directions. They
  should lie along the board and stand up out of it. If they point somewhere
  silly, that view is wrong;
* **a number** — how far that view's corners are from where the fitted lens
  says they should be. Under 0.6 px is good and shown in green; over 1.5 px is
  red and worth a look.

Three things you can do:

**Untick any board** you do not trust and it is left out of the fit. The
buttons across the top do it in bulk: *Best spread* (the automatic choice),
*All* (every board is ticked, and every ticked board is used in the fit),
*None*, and *Drop the worst* (everything more than three times worse than the
middle).

**Click a board** to open it full size and **drag a corner** onto where it
really belongs. When you let go it snaps precisely onto the nearest true
corner, so you do not have to be accurate — get within a few pixels and it does
the rest. Corners you have moved are ringed in red. *Keep the change* keeps
them; *Put them back* undoes it. (The full-size picture is read from the video
again: if the video has been moved or renamed since the scan, it says so
instead of opening.)

**Press *Fit the lens from the ticked boards*** and it refits from exactly what
you chose, then re-scores every picture so you can see whether it improved.
After you untick a board or move a corner, **Next waits for this refit** — the
line beside the button says so — because the lens on the last page must be the
one fitted from the boards you now see ticked. Putting the ticks back as they
were brings Next back too.

**About the automatic choice.** It is not simply "the sharpest ones". A lens
cannot be worked out from boards that are all in the same place at the same
angle, however many there are — the maths has nothing to separate a wide lens
close up from a narrow one further away. So it first picks a **spread**: near
and far, middle and edges, square-on and tilted. Then it fits once, measures
every view against that fit, and throws out the ones that disagree badly —
unless doing so would leave too few, in which case it keeps the spread and
tells you. Whatever it decides, it says why in a line at the top, and you can
overrule all of it.

---

When the wand wizard runs, it uses the attached profiles automatically: those
cameras' points are straightened first, their focal length starts from the
checkerboard's, and the finished calibration keeps the correction for 3D. When
**every** camera has a profile the focal lengths are used exactly as measured;
when only **some** do, the wand refines every camera's focal length (starting
from the checkerboard's where there is one), and the *lens distortion* tick
fits the bending of the cameras **without** a profile. A profile measured at
another picture size than its camera is never used: the wizard's *The cameras*
page shows it as "attached, but NOT used" and says why.

**Digitize it in Kinetrace.** Open the wand recording of every camera as the
cameras of one project (section 10, *Setting it up*), align the offsets, then
in **every camera** track the two wand markers as two landmarks with the **same
names** in each camera — for example `wand A` and `wand B`, the same physical
end under the same name everywhere: **Point ▾ → Ball marker** and click each
ball (for a painted dot or a tape mark rather than a ball, press **N** and click
it instead), rename them (right-click → rename), press **Track ▶**, and correct
where the tracker slips (the annotation click and the **J** key make this
quick). Do the same for the dropped ball over the frames while it falls — give
it a name with *ball* or *drop* in it (`ball`), which is how the wizard
recognises it — and for the floor marks if you use them. A few
hundred frames of the wand spread over the recording is ideal; 30 is the bare
minimum.

**Run the wizard: 3D → Calibrate Cameras with a Wand…** It walks you through
five pages and explains each one:

1. *What this is* — and a check that the project has what it needs.
2. *The wand* — which two landmarks are its ends, the measured length and its
   unit. That unit becomes the unit of every 3D result. Not measured yet?
   Choose *wand lengths* and type 1: every distance then reads in wand
   lengths, and a dropped ball (page 4) tells you how long the wand is in
   metres. The page shows how many frames each camera sees both ends in, and
   how many frames two or more cameras share; under 10 shared frames Next stays
   off.
3. *The cameras* — leave the focal length on **Find it automatically** unless
   you have calibrated these exact cameras before (it works when only some of
   the cameras have a lens profile, too). Tick **Also estimate lens
   distortion** only for wide-angle or action cameras (GoPro-style) that have
   no lens profile, where straight lines look bent at the edges of the picture.
   Under *Lens correction* each camera has a row: **Calibrate…** opens the lens
   wizard for it, **Load file…** takes a saved `.klens.json` or an Argus /
   DLTdv profile, **Remove** drops it, and **Use for all** gives this camera's
   profile to every other camera with the same picture size (identical
   cameras: calibrate one lens, share it). A camera that already has a
   different profile of its own is replaced only if you say yes. A shared
   profile says *the same profile as …* on its row. With a profile on every
   camera the distortion tick is greyed out — there is nothing left for it to
   estimate. Identical cameras still differ a little from unit to unit, so the
   wand fine-tunes the focal length of every camera that shares a profile (a
   camera with a profile of its own keeps the profile's). Measured on a test
   rig, a shared profile then gives distances as accurate as one profile per
   camera.
   Any other landmark tracked in two or more cameras can be used as an extra
   constraint; leave them ticked.
4. *Which way is up* — the dropped object (recommended), the three floor
   marks, or neither. The page chooses the dropped object by itself only when
   a landmark named like one (*ball*, *drop*) really falls in two cameras, and
   fills in the frames of the fall (**Find the fall** looks for them again; it
   also runs when you pick another landmark). Only the frames between the
   release and the moment it lands, bounces or is caught belong there —
   correct them if they are wrong. For the floor marks pick the origin, a point
   along +X, a point towards +Y, and a frame where all three are tracked in two
   cameras (the page suggests one). **Next** stays greyed out until the drop is
   seen by two cameras on at least 4 frames, or all three marks by two cameras
   on that frame. *Neither* keeps camera 1's own axes, which is fine for shapes
   and distances.
5. *Calibrate* — press **Calibrate now**; it takes a few seconds to a minute.
   Then read the report. (*Save calibration files…* on this page writes the
   same files as 3D → Export Calibration.)

**Reading the report.** The top line is a verdict in plain words:

- **GOOD** — use it. The program checked that the recovered wand length hardly
  varies from frame to frame (the *wand score*, under 1 %), that the cameras
  agree with the tracked positions to about a pixel (the *reprojection error*),
  that the wand covered a fair part of every picture, that **every camera**
  kept at least 30 usable observations (and at least half of its own), and
  that no more than 5 % of all the clicks had to be set aside as outliers. A
  camera that contributed only a handful of wand positions, or that lost half
  of them to outlier rejection (usually a sync error), keeps the verdict from
  reading GOOD, and the report names it.
- **USABLE** — it will work, but the notes underneath say what is weak. Usually
  more wand frames in one camera, or the wand kept to one corner of the volume.
  It also reads USABLE when the calibration is sound but the world could not be
  turned upright: the dropped object did not fall freely on the frames given
  (it rested, bounced or was caught), or the three floor marks could not be
  used on the chosen frame. The 3D then keeps camera 1's axes, and the note
  says what to change on *Which way is up*.
- **NOT GOOD ENOUGH** — do not use it. The notes say what to fix: typically a
  wrong wand length, a camera whose wand tracking slipped, two cameras whose
  offsets are not aligned, or one camera that shares almost no frames with the
  others. Fix that, and calibrate again.

Two lines are worth checking by hand every time. **Camera spacing** lists the
distance between each pair of cameras: pace it out or measure it with a tape —
if the report says 3.2 m and the cameras are 4 m apart, your wand length is
wrong. And if you dropped a ball, the **gravity check** compares the fall the
cameras measured with the gravity it must be, in your unit (9.81 m/s², 981
cm/s², 9810 mm/s²): a ratio of 1.000 means the wand length and the frame rate
agree; a ratio far from 1 means one of them is wrong, and more than 5 % off
turns a GOOD verdict into USABLE. If the object did not fall freely on those
frames (or its fall is more than about 20 % away from gravity) the vertical is
not taken from it and the check reads **not used**, with the reason. With the
unit *wand lengths* there is nothing to compare with, so the check tells you
how long your wand is in metres (if the frame rate is right): measure it, then
calibrate again in metres.

**Keep it.** Press **Use this calibration** and it becomes this project's
calibration. Then **3D → Export Calibration…** asks where to save (next to the
project by default) and writes a `.kcal.json` (everything, for your animal
projects: open them, then **3D → Import Calibration…** and choose this file —
no pixel-convention question, the file knows), a `…_dltCoefs.csv` for
colleagues who use DLTdv or easyWand, and, for a wand calibration, the report
as text (`…_report.txt`) for your records. If any camera has a lens
correction, a `…_dltCoefs_README.txt` comes too and the message says so: a
dltCoefs.csv has no room for the correction, so its numbers fit
**lens-corrected** pixels only — raw video digitized with it in DLTdv or
easyWand lands in the wrong place near the picture edges. The `.kcal.json`
keeps the correction. Calibrate again whenever a camera is moved, re-zoomed,
or replaced.

**Using the calibration in another program.** The export dialog also offers,
tick as many as you want (**Tick everything** for all): **Anipose**
(`…_calibration.toml`; rename it `calibration.toml` for Anipose),
**OpenCV / Python** (`…_cameras.yml` or `.json`: each camera's K, distortion,
R and t), **MATLAB** (`…_cameras.mat`: both MATLAB's own fields — 1-based `K`,
`IntrinsicMatrix`, `RadialDistortion`, `RotationMatrix`, … — and OpenCV's),
**Blender** (a script that makes the cameras: run it in Blender's Scripting
tab), the **lens profiles** (an OpenCV `.yml` per camera that has one) and the
**camera offsets** (`…_offsets.csv`). You never convert anything by hand: the
program turns its calibration into each program's pixel and axis conventions,
then checks every file by projecting points of the working volume both ways;
the message says how close they agree (normally within a thousandth of a
pixel). Two things it tells you about when they happen: coefficients from
easyWand or DLTdv often describe a **mirrored** world, which those formats
cannot hold, so the exported world is mirrored in Z (and 3D points exported
with it are mirrored the same way; **3D → Set World Axes…** gives such a
calibration a proper right-handed world instead, with nothing left to mirror); and DLTdv's own lens correction is not
OpenCV's, so an OpenCV lens model is fitted to it and its largest error is
named.

### From several cameras to 3D

Once every camera is tracked (the same landmark names in each), the **3D** menu
turns the pictures into positions in space. An entry chosen too early — with
one video, or before there is a calibration — does not fail silently: it tells
you what it still needs. A calibration covers the cameras it was made for: add
a camera to a calibrated project later and the calibration is kept (in the
saved file too), but 3D needs **every** camera calibrated, so calibrate again,
import a file that includes the new camera, or remove it again (its **×** in
the CAMERAS panel) to get 3D back. Four steps, in this order:

1. **3D → Import Calibration…** (skip this if you calibrated with the wand in
   this project — it is already in place). A calibration is a file that says
   where each camera stands and how it sees: the `.kcal.json` written by the
   wand wizard, or a file from the calibration software you already use —
   DLTdv, easyWand or Argus produce a small `…dltCoefs.csv` (one column per
   camera). If you calibrated in DLTdv8 and saved the project in MATLAB's
   ordinary (v7) format, choose the `…dvProject.mat` instead — it also carries
   the lens correction for GoPro-style cameras, which the plain CSV does not. In
   the dialog, tell the program which column belongs to which camera and how the
   calibration counted pixels. **DLTdv and easyWand** count from 1 with y
   downwards, which is the first choice and almost always right; if your 3D
   comes out wrong by a constant amount, this setting is the first thing to
   check. The choice starts fresh for every file you open (it never carries over
   from the last one). A Kinetrace `.kcal.json` states its own convention, and
   so does an OpenCV-style camera file (below), so for those the choice is
   locked. When
   the file records each camera's picture size, the dialog matches columns to
   cameras by size where it can and asks before accepting a column whose size
   differs from the video's; a `dltCoefs.csv` records no sizes, so match its
   columns in the order your calibration software listed the cameras. It
   refuses to put two cameras on one column. The line under the columns says
   whether a lens correction was found and will be applied — and says so when
   a DLTdv project's lens correction could not be read, or a file's
   distortion lines could not be used. Cameras described the
   **OpenCV way** — a K matrix per camera and the rotation and translation
   between them, as a JSON or a plain text file — are accepted too; because
   such a file usually puts camera 1 at the origin, and a DLT cannot have a
   camera there, the program moves the origin to a point the cameras look at
   and tells you by how much (your 3D comes out in that shifted frame; an
   export of the calibration moves it back). An **Anipose** `calibration.toml`,
   an **OpenCV** `.yml` / `.yaml` / `.xml` / `.json` camera file and a **MATLAB**
   `.mat` with a `cameras` struct are read the same way. The same entry is under
   **File → Import → Calibration…**, beside **3D Points…** (landmarks
   triangulated in Anipose or DLTdv, or by another Kinetrace project — for the
   3D view and the kinematics export), **Camera Offsets…** (a CSV of each
   camera's offset and frame rate, matched by camera name) and **Silhouettes
   (mask images)…** (a folder of black-and-white masks from another
   segmentation tool, named with their frame number, the size of the video).
2. **3D → Estimate Sub-frame Offsets…** The program tries small shifts of every
   camera's timing and keeps the ones where the cameras' rays meet most
   cleanly (the *residual*, in pixels, that it shows before and after). Do
   this after tracking, not before: it needs the tracks to measure anything.
   The search can take a while with many cameras, so a card counts the tests
   as it goes and has a **Cancel** button (nothing changes if you cancel). The
   dialog starts with a verdict. **RELIABLE** means the tracks pin the
   timing down: shifting any camera by half a frame makes the cameras
   disagree noticeably more, so the answer is real. **WEAK** means the
   disagreement barely changes with the timing (slow motion, few shared
   frames, noisy tracks): apply only if the numbers look plausible. **NOT
   SUPPORTED** means the tracks carry no timing information at all — do not
   apply; line the cameras up from a flash or a clap instead. When the landmarks
   simply do not move enough in the window to time anything (a still marker
   looks the same at every offset), it says **flat: the landmarks do not move
   enough to time it** and advises you not to apply: pick a stretch where
   the animal moves. Each camera is
   labelled the same way, and a camera that shares no tracked landmark with
   the others in the window is left exactly as it was and says so. The
   question defaults to *No* unless the verdict is reliable. Offsets that are
   already right (a re-check after applying, or cameras synced by a flash)
   read RELIABLE with "Nothing needs to change". Applying them reconstructs
   the 3D again straight away.
3. **3D → Reconstruct 3D Landmarks** (**Ctrl+3**). Every landmark seen by at
   least two cameras at the same instant becomes an X, Y, Z position in the
   calibration's units (metres if the wand was measured in metres). The **3D
   view** window opens (**3D → Show 3D View**, **Ctrl+5**, opens and closes it
   later): drag to turn it, wheel to zoom, middle-drag or Shift+drag to pan,
   double-click to reset. It follows the frame you are on. The result comes with a verdict. The
   number behind it is the *residual*: for every landmark the 3D point is
   projected back into each camera, and the residual is how far that lands
   from where you tracked it, in pixels — how much the cameras **disagree**.
   **GOOD** (under about 1.5 px on a 1080p picture, scaled for larger ones):
   the cameras agree. **USABLE, with care** (up to about 5 px): check the
   worst landmark the report names, and the camera offsets. **NOT
   TRUSTWORTHY**: a landmark was tracked onto different body parts in
   different cameras, a frame offset is wrong, the calibration's pixel
   convention or camera order is wrong, or the calibration itself is bad — in
   that order of likelihood. With only **two cameras** the residual cannot see
   a mistake that lies along the line joining the two views, so a low number
   is necessary but not sufficient; a third camera makes such mistakes
   visible, and the report says so. The same holds when nearly all positions
   (90 % or more) were seen by only two of your cameras: GOOD then reads
   USABLE. When more than half were, the report names every landmark that no
   third camera ever saw — those have no cross-check at all. If you want the
   axes of the 3D world to be your own (an origin and an X direction you chose),
   **3D → Set World Axes…** does that afterwards: see *Choosing the axes of the
   3D world* below.
4. **3D → Carve Volume at This Frame** (**Ctrl+4**). If the animal is outlined
   (section 4.3) in **three or more** cameras that look at it from clearly
   different directions, the program carves the space every outline agrees
   on — the animal's **volume**, shown as a solid in the 3D view with its
   size in the caption. Two outlines are not enough: they carve a long sliver
   along the line between the cameras whose size is not the animal's, so the
   program declines and says why. A camera that sees the *whole* animal now
   also votes against the space outside its picture, so a tightly framed
   camera no longer leaves a long two-camera sliver in the shape; where a
   camera cuts the animal at its picture edge it cannot rule out what lies
   beyond, and if more than about a tenth of the volume was checked by fewer
   than three cameras the program warns you (*Volume partly checked by two
   cameras only*): the size is then probably too large, so frame the whole
   animal in every camera or carve at a moment when three see all of it.
   Outlines only ever *cut away* space, so
   more cameras give a tighter shape, and a bad outline in one camera bites a
   piece off: check the silhouettes first. It needs step 3 first: the
   reconstructed landmarks tell it where in space to look. If the shape runs
   into the edge of the space searched around them, the program enlarges that
   space and carves again; if it still touches the edge, it warns that the
   volume is cut off (too small) — usually a silhouette that takes in some
   background. **3D → Export Mesh of This Frame…** writes the shape as an OBJ
   or PLY file for Blender, MeshLab or MATLAB.

Most entries of the **3D** menu can be clicked as soon as a video is open (the
two calibration wizards even before): if something is still missing — a second
camera, a calibration — the entry says what, and how to get it, instead of doing
nothing. Three stay grey until they have something to work on: *Show 3D View*
needs a calibration, *Re-track Disagreeing Stretches* a reconstruction, and
*Export Mesh of This Frame* a volume carved at this frame.

**A calibration must cover every camera.** If you add a camera to a project
that is already calibrated, the calibration is kept, in the project and in its
saved file, but 3D waits until every camera is calibrated, and the program tells
you so when you add it. Calibrate again with all the cameras, import a
calibration that includes the new one, or remove the new camera again with its
**×** in the CAMERAS panel to get 3D back. Meanwhile the dashed guides and the ◇
(below) keep working among the cameras the calibration does cover; the new
camera simply gets none.

**Two helpers appear once a calibration is in place.** Loading or making a
calibration switches the first one on.

- **Dashed guides while you place a landmark.** Select a body part (or place one)
  and the program draws, in **every** camera on screen, a dashed line for each
  *other* camera that has that body part at this instant: the line is
  everywhere the part *can* be, given where that camera sees it. So: click the
  snout in camera 1, and camera 2 shows at once where the snout must be; click
  camera 2 — the snout stays selected — and click it on the line (or press
  **N** first, and the one click on camera 2 switches and places). Each line
  runs right to the edges of the picture, or stops where it must: at the point
  that is infinitely far along the other camera's line of sight, or at the other
  camera itself when it is in the picture — the part cannot be beyond either.
  With a strongly curved lens, the stretch where the lens correction is only
  guessing is drawn dotted.
- **The 3D rmse once two cameras have it.** As soon as the body part is placed
  in two cameras at this instant, the program works out its 3D position and
  writes, beside the part in each camera that has it, how well the cameras
  agree: *3D rmse 0.84 px · 2 cams*. It is the reconstruction residual in
  pixels (DLTdv's definition — the number 3D → Reconstruct reports for every
  frame): green = good (under about 1.5 px at 1920 px wide), amber = usable,
  red = one of the placements is off. With only **two** cameras it never reads
  better than amber ("ok"): a slip along the line joining the two views is
  invisible to two cameras, so a low number cannot be called good. The status
  bar says the same after each click. The dashed lines turn faint in every camera from then on — they have
  done their job. With only two cameras a slip *along* the other camera's line
  does not raise the rmse; a third camera does catch it.
- **A ◇ once two cameras agree.** When two or more *other* cameras have the
  part at this instant, their lines of sight meet in 3D, and a **◇** (a small
  diamond) shows where that point is in this camera; the dashed lines fade. Press
  **A**, or click on the ◇, and the part is placed exactly there (hand-placed,
  one **Ctrl+Z**). In a camera where you have already placed it, the ◇ is a
  check: its label says how many pixels your placement is from the other
  cameras'. There is no ◇ when those cameras disagree (one of them is on the
  wrong spot) or see the part along almost the same line (their crossing would
  be a guess) — the line's label says which, and **A** says why in the status
  bar. With two cameras, a slip *along* the other camera's line cannot be seen
  by either: check that stretch by eye.
- **Look here: Alt+click.** An Alt+click on the video edits nothing: it draws
  where that spot can be in the other cameras (a dashed "?" ring marks the spot;
  **Esc** clears it). A plain click with no body part selected places nothing
  either, and says so on the video: press **N** to add a point, or select one in
  LAYERS first.
- **Snapping.** Right-click the part → **Snap to the other cameras' rays here** moves it
  there for you (the nearest point on one line, the crossing of two or more),
  marks the frame hand-placed, and Track re-seeds from it — that is how a
  camera that slid onto the wrong body part gets pulled back into agreement.
  One undo step (**Ctrl+Z**). Two exceptions, both said in the status bar:
  when the cameras stand in a line their lines coincide, so it moves the part
  onto the line only — check its place *along* the line by eye; and when the
  lines cross outside this camera's picture it does nothing (out of the
  picture means no data there). *View → Epipolar guides from the other
  cameras* switches the lines off.
- **A magenta band on the timeline.** After Reconstruct, each lane shows a
  magenta band along its lower edge on every frame where *this camera's*
  position for the part disagrees with the other cameras' by more than about
  5 px (scaled for larger pictures). It is per camera: switch cameras and the
  bands change. A band that starts and stays is a body part that slid onto
  the wrong spot in this camera; go there, snap or re-place it, track again —
  or let the program do it: **3D → Re-track Disagreeing Stretches…** (it
  needs **three or more cameras** — with two, nothing can say which one slid)
  lists every such stretch (which camera, which part, which frames, how far
  off), and on your yes it moves the part onto the other cameras' rays at the
  first frame of each stretch that two other cameras see, re-tracks it through
  the stretch with the ordinary tracker, reconstructs again and reports
  **BETTER**, **NO CHANGE** or **WORSE** with the disagreement before and after
  — you choose to keep it or put everything back (the question defaults to
  keeping only when it is BETTER). **X** or **Space** while it runs stops the
  whole queue and goes straight to that question. After a Reconstruct, a
  message tells you how many such stretches there are. Stretches it will not
  touch are listed as *(skipped)* with the reason: fewer than two other
  cameras see the part, the other cameras place it outside this camera's
  picture, or the cameras disagree among themselves (then the offsets or the
  calibration are the problem). WORSE means either that frames were **lost** —
  the tracker stopped inside a stretch; the message names those frames: undo,
  then click them by hand — or that the other cameras, the offsets or the
  calibration are the problem, not this camera.

**Ctrl+E → 3D landmarks** exports the positions, one row per frame of the
reference camera, with a second file giving each position's residual, how
many cameras saw it, and each camera's own error in pixels. Everything 3D is
saved in the project, so reopening it brings the calibration, the offsets, the
positions and the bands back. The 3D points you export always come in the same
world as the cameras you export with **3D → Export Calibration…**, so the two
can be used together in another program. A DLTdv xyzpts file keeps a frame with
no 3D position as an empty frame, so every later frame stays on its own row.

### Choosing the axes of the 3D world

A calibration puts the origin and the axes somewhere: wherever the wand's
centre was, with a vertical from a dropped ball, or at camera 1 if you gave it
neither. If you want the world to mean something to you — the corner of a
frame as (0, 0, 0), the long edge of a table as the X axis — use **3D → Set
World Axes…**. It needs a 3D result first (**Ctrl+3**), because you pick the
axes **from three landmarks** that were tracked in 3D. A pole's foot, three
marks on the floor or the corners of a frame all work:

1. **Origin (0, 0, 0)** — the landmark that becomes the new origin.
2. **+X points toward** — a second landmark: the X axis runs from the origin
   to it.
3. **+Y points toward** — a third landmark, anywhere on the side where you
   want +Y; the program makes Y exactly perpendicular to X.
4. **+Z follows the right-hand rule** (point the fingers of your right hand
   along +X and curl them towards +Y: your thumb is +Z). That also fixes a
   mirrored world, which easyWand and DLTdv calibrations often have: the new
   one is a proper right-handed world.

Choose the three landmarks' positions **at the frame on screen**, or, for
markers that never move (marks on the floor, a fixed pole), as **their median
position over the whole 3D result**, which is steadier. The window says in
plain words where each landmark lands (for example how far the +X landmark lies
along X) and greys out OK until the three are different landmarks that all have
a 3D position.

What changes: the calibration and the 3D result are re-expressed in the new
world. **What does not change:** the pictures, your tracks and every camera's
pixels (each camera still projects every point to the same pixel as before).
A volume you carved earlier is cleared, because it was in the old axes: carve
it again (**Ctrl+4**). Afterwards **export the calibration and the 3D points
again** to get them in the new world, and then every number you measure (a
speed, a height) is along your axes.

---

## 11. Saving and coming back later

**Your project changes only when you save it.** Press **Ctrl+S** (pressed while
a run is still tracking, it says so and saves the moment the run stops — **X**
stops it at once). The first
time, it asks where to put the project and what to call it: the program makes a
**folder** of that name ending in `.kinetrace` — that folder *is* your project;
**Ctrl+Shift+S** saves it under a new name. If you choose a project folder that
already exists (the save window opens a folder when you pick its name), the
program asks whether to save your work *as* that project — **Yes** replaces what
it holds (its last save is kept in its `.history`), **No** lets you choose
another name; the project that is already open is simply saved. It contains everything — all
positions, the silhouette, your landmark names, events and notes, every camera
and its offset, the calibration and the 3D results, lens profiles, body poses —
and exactly how you left the program: the frame you were looking at, how far you
were zoomed in, the timeline zoom, the side panel, the camera you were working
in and every toggle. Opening it (**File → Open Project…**, **Ctrl+Shift+O**:
go into the project folder and choose its `kinetrace.json`) puts you back
exactly where you were.

**Saving is quick, even for a big project:** a save writes only what changed
since the last one — correct one point and only that point's file is written.

**The save before is kept.** The files a save replaced are kept inside the
project folder, in `.history`: that is your previous save. To go back to it,
close the project and run `python -m kinetrace.convert previous name.kinetrace`
(once: it goes back one save). If the program or the computer stops *while*
saving, the next time you open the project it is put back as it was at the save
before, and the program says so. A save only ever replaces or removes the
files the previous save wrote: **your own files inside the project folder** (a
spreadsheet you made from a table, a note, a copy of a folder) are never
touched, and hidden files the operating system adds (`.DS_Store`, the `._…`
files a Mac leaves on some drives) are ignored. A file in `.history` that is
open in another program does not stop a save.

**Unsaved work is kept safe every 30 seconds** — also while tracking is running
and when you close the program — in a *recovery copy*, never in your project.
So your last Ctrl+S is always there to go back to, and if the program or
the computer crashes you lose at most half a minute. The status bar shows
*Unsaved work kept safe ✓* with the time.

* **Coming back after a crash:** open the project again (or, if you never saved
  one, the video). The program finds the unsaved work and asks *Unsaved changes
  found* (or *Restore unsaved work?*). **Yes** carries on where you stopped;
  **No** opens the last save and keeps the unsaved work aside — nothing is
  deleted. If you open just one camera's video of a multi-camera session you
  never saved, the unsaved work comes back with every camera, each on its own
  video.
* When the program starts and unsaved work is waiting, a notice says so over
  the video: **click the notice** to see the list and restore it. **File →
  Recover Unsaved Work…** shows the same list (everything waiting, newest
  first) at any time.
* **Moving or renaming the project folder does not matter:** its unsaved work
  is found by the project itself, not by its name or where it is.
* If you open an **older copy** of a project while a newer copy has unsaved
  work, the program offers that work as a separate, unsaved copy. It never mixes
  two versions of a project.
* Recovery copies live in the `recovery` folder inside the Kinetrace folder. If
  that folder cannot be written (Kinetrace installed somewhere read-only), they
  go into your own user folder instead, and the program says where, once. Work
  you chose not to restore is moved into `recovery/declined`, work that could not
  be read into `recovery/damaged`; nothing there is deleted automatically.

**Closing** — the window's × or **File → Quit** (**Ctrl+Q**). **With unsaved
changes** it asks *Save changes?* and lists what changed since your last save
(*cam1: P1 (positions)*, *the events* …), so a point a stray click moved is
seen before it is saved (a click on the video with a point selected places that
point). If what changed was put back (Ctrl+Z), it says nothing differs any
more. **Save** writes the
project, **Discard** drops the unsaved work, **Cancel** keeps the program open.
Opening another video or project asks the same question (Cancel then opens
nothing). Closing that question any other way keeps the unsaved work for File → Recover
Unsaved Work…. Moving through the video, zooming and switching toggles are not
changes that need saving: close without saving and the project still opens where
you left it.

**If the tidying after a save has a problem** (a leftover file that could not
be removed, say), the program says *Saved. Not everything could be tidied up:*
and what it was. Your save is complete, and the next save tries again.

**If a save fails** — the folder is gone or read-only, the drive is
disconnected, or one of the project's files is open in another program (a CSV
in Excel, for example) — the program says so (*Could not save the project*) and
the save undoes itself: the previous save is unchanged. Your work is still in the program and in its recovery copy: use
**File → Save Project As…** to save it somewhere else. If your very first save
of a project failed, you can try the same name again: the half-made folder is
no obstacle.

**The project is readable without Kinetrace.** Open the folder: every table
is a spreadsheet-style CSV file that Excel, MATLAB, R or Python open directly —
nothing to unzip. Each camera has a folder (`cameras/cam1`), and in it one file
per landmark (`tracks/snout.csv`: one row per frame that has a position, pixel
positions counted from 0), the landmark list (`points.csv`), events and notes;
each animal's silhouette area and position per frame are in `silhouette/summary.csv`, the
3D result in `reconstruction/points`. Silhouette outlines and body poses are
NumPy `.npy` arrays, which Python, MATLAB and R can read. The folder's
`README.txt` says what every file is. If you edit a file by hand, close the
project in Kinetrace first; the program reads your edit the next time it opens
the project, and a file it cannot read is named in the message (with the row,
for a table). (`.cache` holds copies for fast opening and can be deleted; leave
`.history` alone.)

**One file, to e-mail or archive:** **File → Export Project as One File…**
writes the whole project as a single `.kinetrace` file; **File → Open
Project…** opens it like the folder. A project saved before September 2026 is
such a single file: the first time you save it, the program offers to turn it
into a folder (the single file is kept beside it as `name.kinetrace.bak`), or to
keep saving it as one file.

**Files for other programs, kept up to date:** **File → Keep Exports Up to
Date…** — tick DLTdv8 (one file per camera, or every camera in one file for
3D), DeepLabCut, MATLAB, a wide CSV or the 3D points, and after every save the
program writes them into the project folder's `exports` folder (only the files
whose data changed, in the background). The files your analysis reads are
then always the latest.

A project does **not** contain the video itself, only where the video is on
disk. Keep the videos beside the project folder, or put them inside it in a
folder called `videos`, and move the whole lot together: you can move it
anywhere, even to another computer or operating system, and the program finds
each video again. If it cannot (or the project holds no path for a video at
all), it asks you to point it at each one (*Locate video*). If the video you pick has a different number of
frames from the one the project remembers, it warns you: that is usually another
cut or take, and the tracks would sit on the wrong frames.

**A camera whose video you cannot find** (you press Cancel, or it will not open)
is left out of this session, and the program says so (*Opened without some
cameras*). Your project is **not** changed: it still holds that camera with
every point, track and the calibration, and nothing is saved over it
automatically — Save asks for a new name. To work with every camera, close
the program, make the videos reachable (connect the drive, or move them next to
the project) and open the project again. Without the video of the camera you
were working in when you saved, the project does not open at all.

---

## 12. Getting your numbers out

**File → Export Tracks…** (**Ctrl+E**) opens a save window; pick the format in
its file-type list (*Save as type*).

**With several cameras**, the 2D formats — every row above *ALL CAMERAS* in the
table — export the camera you are working in. The suggested file name carries
that camera's name (`myproject_cam2_tracks.csv`), and the message afterwards
names the camera, so one camera's files never overwrite another's. (The
overlay video's suggested name carries it too.) Click the
next camera's picture and export again for each one.

| Choose this | If you want |
|---|---|
| **Wide CSV** | the general-purpose table — one row per frame, `x`, `y` and a visible flag per body part. Opens in Excel. **Start here if unsure.** |
| **DeepLabCut CSV** | to load your results into a DeepLabCut workflow |
| **DeepLabCut multi-animal CSV** | the same for DeepLabCut's multi-animal projects: one *individual* per animal in LAYERS, and the Scene points as DeepLabCut's unique body parts (under `single`). Needs at least one animal with points |
| **SLEAP analysis CSV** | to load your results into SLEAP or a script written for it: one track per animal, plus a `scene` track for the Scene points; columns `track`, `frame_idx`, `instance.score`, then `<part>.x`, `<part>.y`, `<part>.score` per body part. Needs at least one animal with points |
| **DLTdv8 xypts CSV** | to load one camera into DLTdv8. Written the way DLTdv8 writes its own files: pixels counted from 1, origin at the top-left, `NaN` where there is no data. A DLTdv export is exactly two files: the points file and the `_pointnames.csv` beside it, which names the body parts (the real names, quoted properly even when one contains a comma) and states this |
| **DLTdv xypts CSV, bottom-left origin** | the same for older DLTdv versions and Argus Clicker, which count y from the bottom edge. If a re-imported file lands upside down, you picked the wrong one of these two |
| **Sparse TSV** | only the frames that actually have data — much smaller for sparse tracks |
| **MATLAB (.mat)** | to load straight into MATLAB, including confidence, the silhouette, events and notes. Pixels are counted from 0 like the Wide CSV (the file's `pixel_convention` says so; add 1 for DLTdv8-style coordinates); `NaN` where there is no data |
| **ALL CAMERAS — DLTdv8 xypts** | **the 3D one.** Every camera in one file (`pt1_cam1_X`, `pt1_cam1_Y`, `pt1_cam2_X` …), landmarks matched across cameras **by name**, DLTdv8's pixel convention. One row per frame of the **reference camera** (the first in CAMERAS), starting at its frame 0; the other cameras are read at the same instant through their offsets. A camera that did not film that instant, or has no landmark of that name, gives `NaN`. The `_pointnames.csv` beside it names the landmarks and states the pixel convention, what the rows are, and which video is cam1, cam2 … |
| **3D landmarks** | the reconstructed positions (after *3D → Reconstruct 3D Landmarks*, Ctrl+3), one row per frame of the reference camera, `NaN` where there is no 3D position, with a second file (`_xyzres.csv`) giving each position's residual, how many cameras saw it and each camera's own error |
| **3D landmarks — Anipose / DLTdv xyzpts** | the same positions in the layout Anipose writes (`name_x`, `name_y`, `name_z`, `name_error`, `name_ncams`, `fnum`) or DLTdv's own xyzpts file (`pt1_X` …, one row per frame of the reference camera from 0, `NaN` where none, names in `_pointnames.csv`) |
| **Silhouette outlines** | the animal's outline on every frame as polygons, in a JSON file (pixels counted from 0); with several animals, one file each, named `…_<animal>.json` |
| **Silhouette masks** | one black-and-white PNG per frame with a silhouette (white = the animal), in a folder beside the name you chose (one folder per animal when there are several, `…_<animal>_masks`) — for other segmentation or measuring tools. Many thousands of frames make many thousands of files, so *Everything* leaves this one out |
| **3D kinematics** | **speeds and accelerations.** The 3D positions smoothed, then velocity (X, Y, Z and speed) and acceleration per landmark, plus a report in words. See below |
| **Everything** | every format above that applies, at once, sharing one base filename: the DLTdv8 file once (not the bottom-left variant), *ALL CAMERAS* only with two or more cameras, the 3D files only after a Reconstruct, and the kinematics with **Automatic** smoothing (it does not ask). It runs in the background with a count and a Cancel; if one format cannot be written (a `.mat` open in MATLAB, say) it goes on with the others and lists at the end what failed |

Two extra files are written beside the plain **Wide CSV**, **DeepLabCut CSV**
and **Sparse TSV** exports when they are relevant: your events and frame notes
(`_events.csv`), and a summary of the silhouette per frame (`_segment.csv`).
*Everything* writes them once, beside the Wide CSV. The DLTdv and 3D exports
write only their own files, so a folder of xypts files for DLTdv holds nothing
else. The MATLAB file carries both inside itself.

**How to read the numbers.** Positions are in **pixels** of the original video.
`x` counts from the left edge, `y` counts **downward from the top** edge — this
is the normal convention for images, but it is upside-down compared with a graph,
so remember it when plotting. The Wide CSV, DeepLabCut CSV, Sparse TSV and
MATLAB file count from **0**: the centre of the top-left pixel is (0, 0). The
**DLTdv8** files (one camera and ALL CAMERAS) keep the origin at the top-left
but count from **1**, as DLTdv8 and MATLAB do, so their numbers are exactly 1
larger. Only the **bottom-left** variant flips `y`, counting it upward from the
bottom edge for the older tools that expect that.

**Empty cells mean no data** — the body part was out of the picture, had been
lost (auto-pause cuts a track where it became unreliable), or you marked it
hidden (Shift+X). They are not zeros. The DLTdv, 3D and kinematics files write
`NaN` there instead of leaving the cell empty, because MATLAB reads an empty
cell as 0; the MATLAB file uses NaN too. Make sure your analysis treats them as
missing rather than as a position at the origin.

**3D kinematics.** Velocity is how fast a position changes; acceleration how
fast the velocity changes. Working them out from tracked positions has one
trap: differentiating amplifies jitter — half a millimetre of tracking noise at
240 fps becomes about 18 m/s² of fake acceleration on average (twice gravity),
with peaks three times that. So the positions are **smoothed first**, and the
export asks how: **Automatic** (recommended — the program filters at many
strengths, finds where filtering stops removing noise and starts removing
movement, and picks that cutoff; the report tells you the number and the
reason), a cutoff you choose in Hz, or none. What a cutoff means: movement
slower than half the cutoff keeps 94 % or more of its size, movement at the
cutoff keeps half, and anything faster is removed as jitter. Gaps in a track
stay gaps; nothing is computed across them. Units are those of your calibration
(metres after a wand calibration in metres; "wand lengths" if you did not give
the wand length). Hand checks are in the report: anything in free fall
accelerates at 9.81 m/s²; a landmark that stands still should read near zero
speed.

**What the kinematics report warns you about.** Landmarks that barely move (a
reference marker) are left out of the automatic choice, so they cannot drag the
cutoff down. A landmark that moves faster than the chosen cutoff keeps — a wing
tip beside a slow body — is named in the report and in the file's first line,
with the cutoff to type by hand to study it; if a movement is too fast for the
frame rate to tell apart from jitter, the report says the automatic choice is
uncertain there. With smoothing on, stretches shorter than 12 frames are too
short to smooth: their positions are exported as measured and their velocity and
acceleration are left blank (`NaN`). The peaks in the report leave out one
cutoff period at each end of every stretch, where the filter sees only one side.
The file's first line starts with `#` and states the unit, the frame rate and
the smoothing; in Python, read it with `pandas.read_csv(path, comment='#')`.

---

### A video with the result drawn on it

**File → Export Overlay Video…** writes an MP4 of the camera you are working in,
with everything drawn on the frames: markers and names, the skeleton, the
silhouette, fading trails, the frame number and time, the names of events while
they are running, and your notes. Choose the whole video, an event, the frame
window selected on the timeline, or any range, and tick what to draw — the
**Skeleton bones** tick there decides, whatever *View → Show skeleton bones*
says. Half size is plenty for a talk and renders four times faster (it is the
starting choice for footage wider than 2000 pixels). Rendering runs in the
background — you can keep working, but edits you make while it renders are not
in that video — and the original video is never modified. The new video plays
at the **video file's own frame rate** (a 240 fps camera whose file says 30 fps
gives a slow-motion overlay that plays like the original file, even if you set
its real rate in the CAMERAS panel). **Cancel** deletes the
unfinished file. If the video ends early or a frame is damaged, the message says
where it stopped and how many frames were written. Points you marked hidden are
left out, like in the other exports.

---

### Bringing tracks in from another program

**File → Import → Tracks…** reads points tracked or clicked somewhere else into
the camera you are working in: a **DeepLabCut** CSV (the one it writes when it
analyses a video), a **SLEAP** CSV, a **DLTdv** or **Argus** xypts CSV, or a
landmark's file from another Kinetrace project (`cameras/cam1/tracks/snout.csv`;
it arrives under the file's name) — or an older project's `tracks.csv`. The program recognises the format from
the file itself. If no video is open yet, it asks for the video the tracks
belong to first.

* Points are matched **by name**: a name the camera already has updates that
  point, a new name becomes a new point (and, with several cameras, appears in
  every camera's list). Frames the file has no position for
  keep what they had, so you can import on top of your own work.
* Pixel conventions are converted for you (DLTdv counts pixels from 1, older
  DLTdv and Argus count up from the bottom edge — its `_pointnames.csv` file
  says which). Confidence comes along (DeepLabCut's likelihood, SLEAP's scores);
  instances you labelled by hand in SLEAP arrive as hand-placed.
* A DLTdv / Argus file with several cameras goes into the cameras of your
  project in the file's order (add the videos first); otherwise the program
  asks which of the file's cameras is the one on screen.
* The message afterwards says how many positions were **written** (only what
  actually went in is counted), and how many were left out because they fell
  outside the picture or on frames the video does not have — many of those
  usually means the wrong video. A landmark the file names that Kinetrace
  keeps from the silhouette, or follows as a ball marker, is left as it is and
  **named** in the message. A name that occurs twice in the file becomes two
  points (*name (2)*) rather than being merged. **Ctrl+Z** undoes an import
  into one camera.
* One animal per video: a multi-animal DeepLabCut file or a SLEAP file with
  several tracks is refused with the reason (export one animal per file).

The other **File → Import** entries (calibration, 3D points, offsets,
silhouettes) work the same way: they say what they did, or why not.

* **Camera Offsets…** matches the file's rows to your cameras by name and
  measures every number against your project's **first** camera. If the file's
  numbers were measured against another camera, they are re-based on yours (the
  message says so): a table saying camA 120, camB 97, camC 130 becomes 0, −23,
  10 when camA is your first camera. If the file has no row for your first camera it
  cannot be re-based, and the program asks whether to apply the numbers as
  written.
* **Silhouettes (mask images)…** is all or nothing: if any image in the folder is
  the wrong size or unreadable, nothing is stored and the message names that
  file, so a half-imported silhouette never sits unnoticed in the project.

**Without opening the program.** The same conversions run from a command
window, which is handy for many files at once: `python -m kinetrace.convert`
(with the Python inside the Kinetrace folder: `.venv\Scripts\python` on
Windows, `.venv/bin/python` on Linux and macOS) followed by what to do — for
example `tracks results.csv out.csv --to dltdv --video clip.mp4`, or
`check myproject.kinetrace` to be told whether a file made elsewhere is sound.
`python -m kinetrace.convert --help` lists everything; the online guide's page
*Working with other programs* explains each command. Three things worth
knowing: `convert tracks … --camera 2` refuses an all-cameras file (its rows
are frames of the reference camera; `convert import PROJECT --tracks FILE`
maps them through your project's camera offsets), `convert points3d …
--calibration FILE` writes the 3D points in the same world as the cameras
exported from that calibration, and `convert import … --out NEW.kinetrace`
gives the new project an identity of its own, so its unsaved-work copies are
never confused with the original's.

---

## 13. Measuring a person's joints and joint angles

Everything so far has been about points **you** choose. This section is
different: it finds a **person** for you, works out where all their joints are,
and turns those into **joint angles** — how bent the knee is, how far the hip
has swung, how far forward the trunk is leaning — on every frame.

You do not place any points for this. You do not need a skeleton template. You
point it at a range of frames and it does the rest.

---

### What you get

* **Joints** — shoulders, elbows, wrists, hips, knees, ankles, and (with the 3D
  model) heels, toes, neck and more, on every frame.
* **Joint angles** in degrees, with their rate of change in degrees per second.
* **A side-by-side view**: the footage with the skeleton drawn on it next to the
  pose on its own, with the angles plotted against time underneath.
* **Spreadsheets** of both, and **a report** that says what every angle's zero
  means and how far each joint moved.

---

### The two models, and which to pick

**Body → Find People & Measure Joints…** opens a window whose first choice is
the model. The difference between them matters more than anything else on that
page.

| Model | What it gives | What it needs |
|---|---|---|
| **SAM 3D Body** | Joints **in space** — real 3D from a single ordinary camera, so the angles are true anatomical angles no matter which way the person is facing. 70 joints. | A licence from Meta and a one-off download (below), **and an NVIDIA graphics card**: Meta's code runs only there, so on a computer without one (or on a Mac) the entry is greyed out with that reason. |
| **ViTPose (2D)** | Joints **in the picture** — 17 of them, where they appear on screen. Angles are measured flat, in the image. | Nothing. It downloads itself the first time and needs no licence. |

If the model you want says *code not installed* or *weights not installed*, the
line underneath tells you exactly what to fetch and where to put it. Nothing is
downloaded behind your back.

**Setting up SAM 3D Body** (one off). Meta splits it across three places,
and all three are needed:

1. Ask Meta for access at `huggingface.co/facebook/sam-3d-body-dinov3` and wait
   to be accepted.
2. Download the whole repo into `models/sam-3d-body-dinov3/` — the easiest way
   is `hf download facebook/sam-3d-body-dinov3 --local-dir models/sam-3d-body-dinov3`,
   because it brings **both** parts. `model.ckpt` (2 GB) is the network, and
   `assets/mhr_model.pt` is the body rig that turns what the network predicts
   into an actual body. **The model cannot start without the rig**, and it is
   easy to miss because it sits in a sub-folder.
3. Meta's inference code from `github.com/facebookresearch/sam-3d-body` goes in
   `models/sam-3d-body/` (that folder must contain `sam_3d_body/`), and it needs
   these Python packages:
   `omegaconf yacs roma braceexpand timm pytorch-lightning termcolor`.

The lighter **ViT-H** checkpoint (`facebook/sam-3d-body-vith`, into
`models/sam-3d-body-vith/`) is set up the same way and gives the same outputs.

Restart Kinetrace and the model appears as a normal choice. If anything is
still missing the entry says exactly which piece and where to put it.

---

### What "2D angles" really means

This is the one thing to understand before you trust a number.

A 2D model does not know how far away anything is. It sees where a knee *appears*
on the screen. If the person's thigh is pointing towards the camera, it looks
short, and the knee angle measured from the picture comes out **smaller than the
real one**. Turn the person side on and the same limb reads correctly.

So, with the 2D model:

* **Film the person side on** if you care about knee, hip, elbow or ankle angles.
* Compare like with like — the same camera, the same direction of travel.
* Treat the numbers as a consistent measure of change, not as clinical truth.

With SAM 3D Body none of that applies: the joints are in space, so the angles are
right whichever way the person is facing. The side-by-side view and every
exported file always say which of the two you are looking at, so you can never
mistake one for the other later.

---

### Finding the person

The second choice on the page is how to find the person on each frame.

**Automatically** runs a person detector first. This is the normal choice and it
works on ordinary footage of real people. The first time, the window tells you
that the detector (RT-DETR, 81 MB) will download before the run starts; **Stop**
during that download or while a model loads simply ends the run with a plain
notice.

**Use the segment silhouette already in this view** uses the outline you drew
with the segment tool (**S**) instead. Choose this when:

* the detector misses your subject — unusual clothing, unusual camera angle,
  underwater, a costume, a very small figure;
* there are several people and you only want one of them;
* you already segmented the person for something else.

A frame with no silhouette is **left blank** — it is not filled in with a guess.
In the person list such a run says *from the silhouette* instead of a detector
confidence, because nothing was detected: you told it where the person is.
That is deliberate, and it is the same rule the rest of the program follows.

If you have several people in shot, raise **People** (up to 8) and each one keeps
their own column in the results, followed from frame to frame.

**Use this camera's measured focal length** — a 3D model has to guess how wide
the lens is. If you have calibrated this camera's lens (*3D → Calibrate a Lens
(checkerboard)…*), tick this and the distances in metres come out right. Only
the focal length is used — the frames are not straightened first — and the
angles do not depend on it. Without a lens profile for this camera the box is
greyed out.

If your clip is long, raise **Every Nth frame** to sample it quickly first — the
frames in between stay blank and you can fill them in later.

**Which frames** chooses the whole video, the frame window you selected on the
timeline, or only the frame you are on. **Keep the 3D body shape** stores the
body surface for the side-by-side view (about 110 kB per frame and person).
**Use this camera's measured focal length** is offered when this camera has a
lens profile (*3D → Calibrate a Lens*): the 3D model then knows how wide the
lens is instead of guessing, which makes its distances right. **Body → Remove
Body Pose** deletes every pose in this camera; it asks first, and **Ctrl+Z**
brings it back. Rates of change
(degrees per second) are worked out between the sampled frames, and the
side-by-side video of a sampled run holds only the posed frames, played at the
video's frame rate divided by N so it still runs in real time. The *Which
frames* box chooses the whole video, the window selected on the timeline, or
this frame only.

**Running again adds to what is there.** A later run — over the frames in
between, over a window you selected, or one you stopped half way with **Stop** —
replaces only the frames it actually looked at and keeps every other pose; the
message says which frames were posed again. If a frame it looked at has nobody
on it this time, that frame **keeps its earlier pose** rather than being
blanked, and the message says how many frames and which. With several people,
each person keeps their own column across re-runs: the new run recognises who
is who from the poses already there, so person 1 does not turn into person 2
on the frames you re-ran. A run that cannot be combined with
the earlier poses (the other model, or a different number of **People**) asks
first, because it would replace them all. Ctrl+Z takes back a whole run either
way. If you switch to another camera while a run is going, the result is still
stored in the camera it was started in.

---

### The side-by-side view

**Body → Side-by-side View** (**Ctrl+6**) opens the window this feature is built
around. It follows the video: whatever frame you are on, that is what it shows.

* **Left** — your footage with the skeleton drawn on it. The person's **left side
  is green and their right side is orange**. This is how you catch the classic
  mistake where a model swaps someone's legs over: watch the colours during a
  stride and they should never flip.
* **Right** — the same pose with the picture taken away. With a 3D result you can
  **drag to turn it round** and look at the pose from any direction; double-click
  to put it back. With a 2D result there is nothing to turn, so it stays put — it
  is still useful, because the figure is re-centred and stays the same size even
  as the person walks across the shot.

  Two things about this panel are worth knowing:

  **Stand upright** (on by default) turns the pose so the body's long axis is
  straight up the panel. A pose model reports joints as *the camera* sees them,
  so a camera mounted high and angled down — which is most rigs — reports a
  standing person lying over at the camera's angle. With this on, a standing
  person stands up whatever the camera was doing, and you can compare frames
  and clips without tilting your head. Turn it off to see the raw camera frame.

  **Body shape** draws the actual 3D body surface the model returned, instead of
  a stick figure (drawn the right way out: you see the front of the body, which
  is where the face and the skeleton's front are, in the window and in the saved
  video alike). Only a 3D model produces one, so the box is greyed out for the
  2D model. You choose whether to keep it when you run the estimate (*Keep the
  3D body shape, not just the joints*); it costs about 110 kB per frame per
  person and every save of the project writes it again, so turn it off for a
  very long clip.
* **Underneath** — the joint angles plotted against time, with a line marking
  where you are and the current value of each one on the right. Gaps in a line
  are frames with no data; they are left as gaps rather than joined up. The
  **Plot angles** menu in the window's top bar chooses which angles are drawn:
  a whole group (arms, legs, trunk) or single angles out of all 14, up to **8
  at a time** (ticking a ninth drops the oldest choice). Until you choose, it
  plots the arm and leg flexions. This choice only affects the plot: the
  **Angle numbers** tick and every export are separate.

With several people, pick one in the **Person** list. Tick **Joint names** to
label every joint, untick **Angle numbers** for a cleaner picture, or untick
**Angle plot** to give the pictures the whole window. **Save video…** writes
exactly what you are looking at to an MP4, over the frames that have a pose —
only those inside the window selected on the timeline, if there is one.

---

### Reading the angles

Every angle says what its zero means, in the window, in the report and in the
spreadsheet header. You never have to guess.

| Angle | 0 degrees means | Positive means |
|---|---|---|
| Elbow flexion | arm straight | more bent |
| Knee flexion | leg straight | more bent |
| Hip flexion | thigh in line with the trunk | knee drawn forwards (negative = leg trailing behind) |
| Shoulder flexion | upper arm alongside the trunk | arm swung forwards (negative = behind the body). With the arm raised straight overhead it runs past 180 degrees and goes on, instead of flipping to −178 |
| Ankle dorsiflexion | foot square to the shin | toes pulled up (negative = toes pointed) |
| Trunk lean | this person's own average posture over the clip | leaning forwards |
| Neck flexion | ears straight above the shoulders, in line with the trunk | head carried forwards. It is measured to the ears, so a nod of the head alone barely changes it |
| Thigh separation (stride) | thighs side by side | left knee ahead of the right (negative = right knee ahead) |
| Shoulder–hip twist | shoulders square over the hips | shoulders turned one way about the trunk (3D only) |

Hips, shoulders and ankles are **signed** — they can go negative — because those
joints swing both ways and a number that could not tell flexion from extension
would be useless for a walk. The neck, trunk lean, thigh separation and the twist
are signed too. Knees and elbows only fold one way, so they are never negative.

**Which way is forwards, in a flat 2D picture.** A signed 2D angle needs to know
which way the person faces. The program takes it from the nose and toes where
they are seen; on a frame where the nose is hidden it uses the way the person
**usually faces** over the clip, and where even that is unclear it leaves the
signed angle **blank** rather than guess a sign that could be backwards. (With
the 3D model none of this matters.)

**Trunk lean is measured against the person, not the camera.** Its zero is that
person's own average posture across the clip, so a camera mounted high and
angled down cannot add a lean that is not there. If you want lean against true
gravity you need a calibrated rig — that is what the 3D menu's wand calibration
is for.

**Ankle angle uses the foot, heel to toe** — not the ankle-to-toe-tip line,
which on a real body sits at an awkward angle and made the number jump.

**A joint the model is unsure of gives no angle.** The 2D model scores every
joint; a joint scored under 15 % — an ankle cut off by the edge of the picture,
say — is treated as unseen: it is not drawn, and any angle that needs it is
blank on that frame. SAM 3D Body gives no per-joint score, so a joint hidden
behind the body is a guess that looks as sure as a visible one — check such
frames by eye.

---

### Checking it by hand

Before you use any of these numbers, spend two minutes on this. It is the same
habit as checking a tracked point.

1. Open the side-by-side view and step through a few frames with **F**.
2. **Do the joints sit on the person?** Watch the elbows and knees in particular.
3. **Do the colours stay put?** Green should stay on the same leg all the way
   through a stride.
4. **Pick one frame and check one angle by eye.** If the plot says the knee is at
   40 degrees, does the leg in the picture look about 40 degrees off straight? A
   protractor on a printout is a perfectly respectable way to settle it.
5. **Look at the gaps.** Frames with no data are blank in the plot and blank in
   the spreadsheet. If there are more than you expected, the model was not finding
   the person — try the silhouette route.

---

### How much to trust each person

A pose model hands back a convincing body for **any** box it is given, so if the
person detector latches onto a tripod you get a confident-looking skeleton of a
tripod. The person list and the report therefore both show how sure the
*detector* was — "person 2 (51 frames, found 92%)". Anything under 60% is
flagged in the report as *check this is really a person*. When several people
are listed, pick the one with the high number.

### Getting the numbers out

| Menu item | What it writes |
|---|---|
| **Body → Export Joint Positions…** | one row per frame and person: every joint's `x`, `y` in pixels (origin top-left, `y` down) and the model's score for it (`conf`; blank for SAM 3D Body, which gives none), plus X, Y, Z in metres when the model gives 3D. The `xyz_frame` column says what those are: `camera` = measured from the camera (X right, Y down, Z away from it), so they show movement from frame to frame; `body` = centred on the body, fine for angles and segment lengths but not for movement. The first lines start with `#` and explain every column — in Python, read it with `pandas.read_csv(path, comment='#')` |
| **Body → Export Joint Angles…** | every angle in degrees and degrees per second, with the frame number and the time in seconds — **plus a report file** beside it (`…_report.txt`). The `#` lines at the top say what each angle's zero means |
| **Body → Export Side-by-side Video…** | an MP4 of the view described above, over the frames that have a pose (inside the timeline's selected window, if there is one); a sampled run gives only its posed frames |

The report is worth reading. It says which model was used, how many frames had a
person on them, whether the angles are 3D or flat, **how far each joint actually
moved** (its range of motion), what every zero means, and a one-line verdict:

* *good* — a person was found on almost every frame;
* *ok* — found on most frames, so check the gaps on the timeline;
* *poor* — found on a minority of frames, so do not rely on it yet;
* *nothing found* — no person anywhere in the range.

**Blank cells mean no data.** They are not zeros, exactly as everywhere else in
this program.

**Body → Remove Body Pose** deletes every pose in this camera; it asks first, and
Ctrl+Z takes it back.

---

### When it goes wrong

**"No person was found anywhere in that range."** The detector did not recognise
your subject. Draw round them with the segment tool (**S**) and run again with
the silhouette option.

**The skeleton is on the wrong person.** Raise **People** so everyone gets their
own column, then pick the right one in the side-by-side window; or segment the
one you want and use the silhouette.

**Left and right keep swapping.** This is a known weakness of single-camera pose
models when a person is side on. It shows up immediately as the green and orange
flickering between legs. The 3D model is much better at it than the 2D one.

**The angles look too small.** Almost always the 2D model on a person who is not
side on — see above.

**It is slow.** Every frame goes through a neural network, so this is much slower
than scrubbing. Use **Every Nth frame** to survey a long clip, or select a window
on the timeline and do only that.

---

### It does not replace the point tracker

The two are for different jobs and they live side by side in the same project:

* This section measures **a human body** with a model that already knows what a
  human looks like. It is fast to set up and you place nothing.
* The point tracker follows **whatever you point at** — a marker on an iguana, a
  fin, a wingtip, a spot of paint — with the accuracy described in section 7, and
  feeds the 3D reconstruction in section 10.

Use the body layer for human movement; use the point tracker for everything else,
and for the cases where you need subpixel accuracy on a landmark you chose
yourself.

---

## 14. Choosing the settings that matter

The defaults are good. These are the ones worth understanding. The switches sit
in the bottom bar with their names beside their icons; on a narrow window only
the icons show — hover over one to see its name. They are yours, not a camera's:
switching cameras leaves them as they are, and a project remembers them.

**Auto-pause** (bottom bar, on by default) — stop as soon as a body part is
genuinely lost (an ordinary moment behind something does not trigger it). The
program jumps to the frame where it happened, selects the point, and **cuts its
track at the first unreliable frame** (red on the timeline), so the data ends
exactly where the tracking stopped being trustworthy. Click the point where it
really is and press Track. Leave it on while you are learning. Turn it off for
footage with a lot of hiding-behind-things, where you would rather review
afterwards than be interrupted.

**Keep its points on its silhouette** (not a switch in the bottom bar: a
setting of each animal — right-click the animal in LAYERS; **off** until you
tick it; it needs a silhouette) — keeps that animal's landmarks honest about
its outline. A landmark tracked by appearance that wanders a few pixels across
the outline (the outline itself jitters from frame to frame) is nudged back
onto the silhouette. A landmark that actually **leaves** the silhouette **stops
the run at that frame**: the program does not pull it back onto some other
part of the animal and carry on, because the spot it would land on is not the
spot you clicked. Its track ends on the last frame it was on the body; click it
where it really is and press Track. To let one point of the animal off its
outline, right-click it → **May leave its silhouette (free point)**. Points in
**Scene** (reference markers on a rock, wand ends) are never held. When a run
stops this way, the message names the point and the frame. An animal with this
ticked shows a small dot inside its swatch in LAYERS. With it ticked,
selecting one of its points brings the animal's silhouette into the run even
if the animal's own row is not selected: without the outline there would be
nothing to hold the point to.

**ROI** (bottom bar, on by default) — when your animal is small in a big frame,
the program works on a zoomed-in crop so it can see more detail. It decides for
itself whether that helps (only when the crop is at least twice as close).

**Follow** (bottom bar, off by default) — **the view stays exactly where you put
it** unless you switch this on: nothing pans or zooms on its own, in any camera
view, and **R** always fits the whole picture. Switched on, it keeps the action
in view while you are zoomed in: with a point selected it pans to keep that point
in view (and holds still where its track has a gap); with nothing selected it
frames everything being tracked, including the silhouette, and re-frames as the
animal moves.

**Point model** (the **▾** arrow on the Track button) — which method follows the
points you clicked. **AllTracker** is the default and holds on to body parts
better over long stretches. **CoTracker3** is about twice as fast and slightly
sharper on high-contrast marks. **Moving spot** is only for a target small
enough to be one point — a dot with no visible shape (see *Which point model
should I use?* at the end of this section). If a track keeps drifting, try another one. The choice applies to the
next run and can be changed at any time. It sets the tracker of the points
selected at that moment (and the default for points that have none of their
own); each point keeps its own, shown as AT / CT / MS on its row, and is saved
with the project. Right-click a point's row → *Tracker* changes just that one.

**Segmentation model** (the **▾** arrow on the Segment button) — which method
draws the outline: **SAM 2.1 base+** (fast, a 617 MB download), **SAM 2.1
large**, or **SAM 3** (the best outlines, about half the speed; Meta must grant
you access first). Each entry tells you whether it is ready to use, needs
downloading, or needs a token. Better outlines cost speed. A new choice applies
to the next silhouette or tracking run.

**Settings** (**Ctrl+,**, or the last entry of that same **▾** menu) holds four
things: **Annotator (your name)**, recorded on every event and note you add; the
**Segmentation model** again; the **Hugging Face token**, needed only for SAM 3 —
ask for access at `huggingface.co/facebook/sam3`, create a *read* token in your
Hugging Face settings, paste it and press **Save token** (it is kept inside the
program's folder, in `models/hf/token`, and is sent only to Hugging Face when
the weights download); and **Mask
opacity**, how strongly the silhouette is tinted over the video.

**Marker px** (bottom bar) — how big the markers are drawn. Shrink them when a
marker hides the very thing you are trying to see.

**Mask** (bottom bar, on by default) — shows or hides every silhouette on the
video; the checkbox on an animal's row in LAYERS shows or hides just that
animal's.

**Settings** (**Ctrl+,**, or the last entry of the **Segment ▾** dropdown) —
your name for events and notes (*Annotator*), the segmentation model, a
**Hugging Face token**, and *Mask opacity* (how see-through the silhouette is
drawn). The token is needed only for SAM 3, whose files Meta releases on
request: ask for access at `huggingface.co/facebook/sam3`, create a *read* token
in your Hugging Face account settings, paste it and press *Save token*. It is
stored inside the program's folder.

### Which point model should I use?

**The rule in one line: can you see its shape?** If you can see a body, a
head, legs, wings or an outline — anything you could put more than one landmark
on — use **AllTracker** (with **Segment** for the outline). If the whole target
is a **dot small enough to be one point** (a few pixels to about 20 across, no
visible shape), use **Moving spot**.

Three point models follow the points you click. The choice is in **Track ▾**
(and **Help → Which Point Model Should I Use?** brings you here) and on each
point's row (right-click → *Tracker*). **Every point has its own**, shown as
AT / CT / MS beside its name and saved with the project, so one project can mix
them: Track tracks each selected point with its own tracker.

| Your footage | Use | Clicking needed |
|---|---|---|
| An animal, a person or an object whose shape you can see — even a small one, if you can make out its body | **AllTracker** (the default), with **Segment** (SAM 3) for the outline | one click per body part to start |
| Painted dots or stickers on a body, with strong contrast, when speed matters | **CoTracker3** | one click per point |
| A target that is only a dot — small enough to be one point (a few pixels to about 20 across), with no shape of its own — often over a background that moves (waves, ripples, sky, leaves): a squid's head spot from a ship, a distant bat, bird or insect | **Moving spot** | the dot on two frames in a row to start; 20 frames in a row for the test below |
| A round marker such as a wand ball | **Point ▾ → Ball marker** (whatever the point model) | one click per ball |

Moving spot follows **one point per target**. On anything bigger it would
follow only the centre of a blob — if you want the head *and* the tail, or the
outline, that is AllTracker's job. The test below says so when the target
measures more than about 20 pixels across.

**Why AllTracker can lose a small spot.** AllTracker and CoTracker3 follow the
*texture* around what you clicked. A squid's white head spot filmed from a ship
is a few pixels across and has no texture of its own; the water around it has
plenty, so they follow the water. Their confidence stays high, so nothing warns
you: on two flying squid both were off within 2 frames.

**How Moving spot works.** On every frame it looks for the spot where its speed
says it should be, with one of three detectors: the brightest small blob there
(*bright spot*: white or pale on water, sky or leaves), the darkest (*dark
spot*: a bird or bat against the sky), or what changes much more than that
background usually does (*unusual change*: a bat over a river). It never
guesses: when the spot is not where it should be, or two candidates look alike,
it **stops on that frame**, the track ends on the frame before, the point is
selected, and you click it again and press Track. A spot that flies out of the
picture simply ends. It needs no graphics card. Regions are left out of a
Moving spot run (they need AllTracker or CoTracker3).

**Before you track with it**, click the spot on two frames in a row (select
it, click it, press **F**, click it) and go back to the first: that gives it
its speed. With a single click its first search is wider, and it stops if two
spots there look alike.

**Let the test decide** — **Track ▾ → Test the point models on my clicks…**, or
right-click the point → *Test the point models on its clicks*:

1. Select the point (or press **N** and click it), then on each frame click it
   where it is; **F** steps one frame. The test needs at least **20 frames in a
   row** (skipping one or two is fine); 30 is better. A Track run over those
   frames replaces them, so click them where you will not track over them, or
   test before you track.
2. Press **Run the test**. It starts every point model from your first click
   and follows your clicks; wherever one drifts away from a click, or stops, it
   is put back on that click (what you would have had to do) and that counts as
   one **correction**. Moving spot is tried at several settings: the search
   widths come from the spot's size and from how much it turns between your
   clicks, so a dodging animal is offered a wide enough search. They take
   seconds. A target measuring about 60 pixels across or more is told that
   AllTracker with a Segment suits it better. Ticking
   AllTracker and CoTracker3 re-tracks them from every correction: a minute or
   two.
3. Read the table and the sentence under it. *Silent drifts* are corrections
   where a model was off without saying so; the rest are stops, where it said
   so. Fewer corrections win; on a tie AllTracker is recommended, because it
   needs no extra clicking.
4. The button reads **Use … for** *your point* (for example *Use Moving spot
   (bright spot, 6 px) for P1*) and gives the winner to **that point only**: its
   row then shows AT / CT / MS, and the project's default and your other points
   are not changed. For Moving spot it also keeps the settings it found on that
   point. Ctrl+Z takes it back.

The program suggests this in two situations: when you have corrected a point
by hand on 5 of the last 20 frames, and when a new point sits on a small,
isolated spot in a camera that has no silhouette. Click the notice to act on it.

---

## 15. When something goes wrong

**The first start stopped before the window opened.** Almost always the
internet connection: start the launcher again, it carries on where it stopped.
If it stops twice, the last lines in the black window say why — copy them into
your question. Everything the launcher fetches goes into the `.venv` folder
inside the program folder; deleting that folder and starting again gives a
clean retry and loses none of your work.

**The program will not start / an error mentions "encodings" or "103".**
The folder was moved or renamed while the program was set up in it. Run
`run.bat` (Mac: `Kinetrace.command`, Ubuntu: `./run.sh`) once; it repairs
itself.

**A red notice says "Something went wrong", or the program closed by itself.**
Kinetrace keeps a record of every error it meets — what went wrong, where in
the program, and what you were doing (the frame, how many cameras) — in the file
`kinetrace.log`, in the `logs` folder inside the program folder. Open **Help →
Error Report…**: it shows the recent errors and the System Check; **Copy** puts
all of it on the clipboard for an e-mail or a bug report, and **Open the log
folder** shows the files if you would rather attach them. If the program closed
by itself, start it again and open the Error Report: the session marked "did
NOT end normally" is the one that closed, and the stack dump under it says
where. Nothing in these files is ever sent anywhere (they can contain the names
of your video files). The notice appears at most once every ten seconds even
when several errors come together; the log keeps all of them. Two Kinetrace
windows open at once share the same log without trouble.

**The status bar says *cpu* although the computer has an NVIDIA card.** If the
speed test of the graphics card failed once (a driver hiccup), the program
tries it again at the next start: there is no file to delete. Open
**Help → System Check…**: it says whether the card's driver is too old for the
program's PyTorch (update the driver at nvidia.com/drivers, then restart the
computer) or whether the CPU version was installed (delete the `.venv` folder
and start the launcher again with the driver up to date). The same report
prints in a terminal with `run.bat --check` / `./run.sh --check`.

**"… could not be downloaded" when you press Track or click an animal.** The
first use of a model downloads it (see section 2). The message says why: no
connection to the server, the connection stalled (the download gives up after
30 seconds without data instead of waiting for ever), or the server refused. Check
the connection (a university network may need its proxy settings), then press
Track again: what already arrived is kept. **"… is not the expected one (its
checksum differs)"** means the file that arrived was not the exact file this
version of Kinetrace was tested with; it was deleted, not used. Try again later,
and if it keeps happening, update Kinetrace (Help → Check for Updates…).

**A model entry is greyed out with "needs an NVIDIA GPU".** That model
(SAM 3D Body) cannot run on this computer; use the 2D model next to it. Nothing
is wrong.

**Everything is very slow.** With no graphics card the models run on the
processor, several times slower; **Help → System Check…** confirms it. A
smaller video, a shorter frame range, or the *Every Nth frame* setting of the
body dialog all help; so does a computer with an NVIDIA card.

**"Could not open video".** The file uses a format the program cannot read.
The error message gives you an exact `ffmpeg` command that converts it.

**The lens profile did not reach my experiment.** The lens wizard gives the
lens it measures to one camera of the project that is open. Two ways to do it:

- **With your experiment open** (the project with your cameras, or that
  camera's own video): choose **3D → Calibrate a Lens (checkerboard)…**. On the
  page *The checkerboard video*, press **Choose…** and pick the video of the
  board. The box starts out showing the camera's own video, which has no board
  in it. Say which camera the lens belongs to (the board video must have the
  same picture size as that camera), press *Find the boards*, and on the last
  page press *Attach to …*.
- **With nothing open yet:** run the wizard anyway and on the last page press
  **Save lens file…**. Later, open your experiment, choose **3D → Calibrate a
  Lens (checkerboard)…** again and press *I already have a lens file…*: it goes
  straight to the last page, where *Attach to …* gives the lens to the camera
  (or press *Load file…* next to that camera in the wand wizard). A lens
  attached only to the checkerboard video's own project stays there and never
  reaches your experiment.

Most 3D entries behave the same way: rather than being greyed out, they open
and tell you what is still missing — a video, a second camera or a
calibration.

**A warning about "variable frame rate".** Your camera did not record at a
steady rate, so frame numbers do not correspond to time reliably. The message
includes the command to fix it. Do this before trusting any results.

**"This file does not state a usable frame rate…".** The file carries no
usable frame rate. The program then either **measured** one from the frames'
own time stamps ("…so Kinetrace measured N fps from its frame timestamps") or,
when even that is impossible, is **assuming** 30 fps ("…so Kinetrace is
ASSUMING 30 fps"). Frame numbers and the tracking itself are not affected.
Everything in seconds is: times, speeds and accelerations, the sound sync and
the ×n rate between cameras. Compare the number with the rate your camera was
set to. If it is wrong, click the camera's **fps button** in the CAMERAS panel
and enter the right rate; nothing needs re-encoding. High-speed files that do
state their rate are read as they are, up to 100 000 fps.

**"This file says it has N frames but only M can be decoded."** Some cameras
write a frame count into the file header that is a few frames longer than
the video really is (a GoPro clip cut short, a recording stopped mid-write).
The program checks by actually reading the last frame and counts what exists;
the missing frames never had a picture, so nothing of yours is lost. If a
project saved earlier used the longer count, its last few timeline columns
simply stay empty.

**The picture looks stale, frozen, or garbled.** Press **Shift+C**. This throws
away the stored pictures and re-reads the current one from the file. It never
touches your tracked data.

**"Tracking stopped: frame N of the video could not be read just now — press
Track again."** One frame failed to read, but the program tried it a second
time and it worked: the file is fine (a busy network drive, a hiccup). Nothing
is lost; everything tracked before that frame is kept. Press **Track ▶** again
to go on.

**"Tracking stopped: frame N of the video could not be decoded."** Even a
second try fails: the video file is damaged at that frame (a copy that did not
finish, a card error), or ends earlier than it claims. Everything tracked up to
the frame before it is kept and saved, the playhead waits there, and a window
(*Damaged frame in the video*) explains. Scrubbing onto such a frame shows it
blank with a message ("frame N could not be decoded … The frame shows blank;
Shift+C retries."). Copy the file from the camera card again if you can. To go
on with this one, go to a frame after the damaged one, place the points there
again (select each in LAYERS and click it on the video) and press
**Track ▶**; or make a clean copy with the command the window gives (`ffmpeg -i
"clip.mp4" -c:v libx264 -crf 18 fixed.mp4`) and open the copy, which starts as
a new, empty session, so do this before tracking much.

**Keyboard shortcuts do nothing.** The keys work wherever the keyboard focus
happens to be — a number box, the LAYERS tree, the right panel even when you
have floated it as a window of its own — except when:

- you are typing a name (renaming a point or an animal): letters are text
  until you press **Enter** or **Esc**;
- another window is in front — a dialog, this manual, the body side-by-side
  view: click the main window first;
- no video is open yet: there is nothing for the keys to act on (**H**, for
  instance, does nothing until a video is open);
- tracking is running: only **X**, **Space** and **T** (all three stop it) and
  the view keys work; **N**, **S** and frame stepping wait until it stops.

In a number box, the keys a number is typed with — digits, **+** and **−**, the
decimal point, **Enter**, **Delete** — stay with the box.

**The buttons at the bottom lost their names, or the window does not fit.**
The window opens sized to your screen. When the bar at the bottom is too
narrow for every label, the tool buttons give up their names one at a time,
least-used first (Pan, ROI, Mask, Follow, Auto-pause, Segment, Point), and
show only their icons; hover over one to see what it does. The names come back as soon as there is room: widen the window, or hide
the right panel with **Ctrl+1**. The Track button always keeps its words.

**The outline keeps grabbing the wrong thing.** Go to the frame where it goes
wrong, press **S**, **Shift+click** the part that should not be included, then
press **Track ▶**. The correction applies from that frame onwards.

**A landmark sits on the wrong end of the animal** (tail where the head should
be). The skeleton's head landmark was not placed, or has slid off the head.
Place it — `snout`, `nose` or `beak`, depending on the skeleton — on the frame
where it goes wrong, then track again.

**You wanted one wing but got the whole bird.** The outlining model works on
whole objects: it answers "which object is here?". Track the wing with named
points instead (wing tip, wrist, shoulder), which is what the skeletons are for.

**Left and right feet (or wing tips) are the wrong way round.** Feet and wing
tips found from the outline are named as seen **from above**, looking down on
the animal's back. Filmed from below, left and right swap: rename the two
landmarks to swap their names (right-click → *Rename point*), or give each the
other side's rule (right-click → *Data source* → *From silhouette: …*; it asks
first, because that erases the point's track, and **Ctrl+Z** brings it back),
then track again. Projects tracked before 22 September 2026 on footage filmed
from above have these pairs swapped: swap their names, or track again, before
exporting.

**Tracking is slow.** Check the bottom bar: if it says CPU rather than a graphics
card, that is why. Otherwise, high resolution simply costs time — 4K runs at
roughly 15–20 frames per second on a good card with the CoTracker3 point model,
and about half that with AllTracker, the default (**Track ▾** chooses). An
outline adds its own time on top.

**It tracked only one body part when you wanted all of them (or nothing at all).**
Track follows exactly what is selected in LAYERS: a point row is that point, an
animal row is its silhouette and all its points. Press **Ctrl+A** to select
every animal and every point, then Track again. The Track button always says
what it is about to do (*Track · 3 points + silhouette ▶*); with nothing
selected it tracks nothing and a notice tells you to select something.

**The silhouette was not tracked along with a point.** A point selected on its
own is tracked on its own. Select the animal's row too (or the animal row
alone, for the silhouette and all its points), or tick *Keep its points on its
silhouette* on the animal (right-click it in LAYERS).

**A new point landed in the wrong animal (or in Scene).** Drag it in LAYERS onto
the right animal, or right-click it → **Move to ▸**. To steer the next one,
select the animal's row before pressing **N**.

**S does nothing when I click the video.** With several animals, **S** needs to
know which one to outline: select that animal's row in LAYERS (one row), or
press **＋ Animal** for a new one. A notice over the video says so.

**Speeds, times or camera matching are off by a round factor (8×, 33×…).**
The file probably states a slow-motion playback rate instead of the rate the
camera filmed at. Click the camera's **fps button** in the CAMERAS panel and
enter the real rate.

**You lost work.** Try **Ctrl+Z** (one step only). Otherwise open the project
again (**Ctrl+Shift+O**) or, if you never saved one, the video: unsaved work from
the last 30 seconds before a crash is offered (*Unsaved changes found* /
*Restore unsaved work?*). **File → Recover Unsaved Work…** lists everything
waiting. Work you once declined is in the `recovery/declined` folder inside the
Kinetrace folder, and work that could not be read in `recovery/damaged`: rename
such a file to end in `.kinetrace` and open it with File → Open Project…. The
save before your last Ctrl+S is kept in the project folder's `.history`:
`python -m kinetrace.convert previous name.kinetrace` goes back to it.

**"Could not save the project".** The save undid itself, the previous save is
unchanged, and your work is still in the program (and in its recovery copy).
The folder may not exist or may be read-only, or one of its files may be open
in another program (close it there and save again): or use **File → Save
Project As…** to save somewhere else.

**"Opened without some cameras".** A camera's video was not where the project
remembers it, and it was not located. That camera is left out of this session
only: the project still holds it, with all its points, tracks and the
calibration, and nothing is saved over it automatically (**Save** asks for a
new name). Close without saving, make the videos reachable (connect the
drive, or move them next to the project), and open the project again.

---

## 16. Glossary

**Animal** — one individual animal or object in the LAYERS panel, with its
points underneath. It may have a silhouette (drawn with **S**) or not. Its
points are named "*animal part*" ("squirrel snout") in exports and in every
camera.

**Auto-pause** — stopping tracking automatically when a body part is lost. The
lost part's track is cut at the first frame where it became unreliable, so what
is left can be trusted. (When the animal keeps its points on its silhouette, a
part that leaves the outline stops the run the same way.)

**Ball marker** — a round marker (a wand ball, a reflective dot, a dropped
ball) found in every frame by outlining it and fitting a circle; the circle's
centre is the tracked point. Added with **Point ▾ → Ball marker**.

**Confidence** — how sure the program is about a position. Low confidence is
drawn red on the timeline.

**Derived landmark** — a body part computed from the silhouette's shape rather
than followed by appearance. Shown with a dotted square in LAYERS and
a dotted ring with an italic name on the video. Needs no clicking, and cannot
be placed by hand: to place it yourself, right-click it → *Data source* →
*Track by appearance*.

**Event** — a named stretch of frames you mark for your own reference.

**Export** — writing your results out as a table other software can read.

**Frame** — one still picture of the video. Numbered from 0.

**Hidden (marked by hand)** — your own statement, with **Shift+X**, that a body
part is not really visible on a frame. Its position stays in the project but is
left blank in every export and out of 3D. Drawn as a ring with a cross.

**Landmark** — a named body part being measured.

**Layers** — the right panel's tree of everything you track: each animal with
its points, then Scene. Track tracks what is selected there.

**Moving spot** — the point model for a target small enough to be **one
point**: a dot with no visible shape (a squid's head spot on water, a distant
bat against the sky). Anything whose shape you can see goes to AllTracker. It looks for the
spot where its speed puts it — the brightest or darkest small blob there, or
what changes much more than that background usually does — and stops where it
cannot tell. Chosen in **Track ▾**; *Test the point models on my clicks* says
whether it suits your footage.

**Offset** — for several cameras: the frame a camera shows when the reference
camera is at its frame 0. A camera switched on 12 frames after the reference
has offset −12; one switched on earlier has a positive offset. The first camera
loaded is the reference and its offset is always 0.

**Point** — something being tracked. Each landmark is a point.

**Point model** — the method that follows the points you click: AllTracker (the
default), CoTracker3 or Moving spot. Chosen in **Track ▾** and saved with the
project.

**Project (`.kinetrace`)** — a folder holding all your work on a video, or on
all the cameras of one recording, and how you left the program (also
exportable as one file). Not the videos themselves.

**Region** — an outlined area (circle, rectangle or polygon) tracked as a whole
and reported as its centre, rather than a single spot.

**Scene** — the LAYERS row for points that belong to no animal: wand ends,
reference marks on a rock, calibration points. They are never held on a
silhouette.

**Segment / silhouette** — the outline of an animal against the background,
found in every frame once you click the animal with the Segment tool (**S**).
Optional: points track without it.

**Skeleton** — a ready-made named list of landmarks for a kind of animal, put on
one animal; it may include bones (lines between landmarks) and a head.

**Tracking** — following something automatically from frame to frame.

**Visible** — whether the tracker judged a body part could be seen in that
frame (a filled marker) or was covered for a moment (a hollow ring). A covered
body part keeps its estimated position, and it is exported.

**World axes** — the origin and the X, Y and Z directions of the 3D positions.
A calibration chooses them; **3D → Set World Axes…** lets you choose them from
three landmarks instead (Z follows the right-hand rule).

---

## 17. Keyboard and mouse reference

The same list is in the program itself: **Help → Keyboard & Mouse Reference**.
The keys work wherever the keyboard focus is — a number box, the LAYERS tree,
the right panel even when floated as its own window — except while you type a
name. Most need a video open, and while tracking runs only stopping and the
view keys work. Hover over any button or menu entry to see what it does.

### Moving around

| Key | Does |
|---|---|
| **F** / **B** | one frame forward / back |
| **Shift+F** / **Shift+B** | jump by the step size set at the bottom |
| **←** / **→** | one frame back / forward (with **Shift**: by the step size) |
| **Home** / **End** | first / last frame |
| **Space** | play / pause the preview |
| click or drag the timeline | go to that frame |
| hold the left button and move (on the video) | pan the zoomed-in view — also when you start on a marker; markers are never dragged, so a drag can never move a point by accident |
| type in the frame box | go to that frame number |
| **O** | onion skin: ghosts of the previous / next frame |
| **L** | loupe: magnifier under the cursor |
| View → Trails / Display filter | fading trails (Off / Last 10 frames / Custom…) and upcoming path; contrast / brighten / frame difference (display only) |

### Working on the video

| Key | Does |
|---|---|
| **N** | arm the crosshair — the next click places a point |
| click (armed) | place a point, or continue the selected one where it has no data |
| drag a circle (armed) | track a whole region as one point |
| click (not armed) | place the point selected in the list here, by hand (on or right beside the ◇: exactly at the ◇); nothing selected = nothing placed (a click on a marker just selects that point). Two quick clicks are two clicks, so frame-by-frame digitizing never loses one. A landmark derived from the outline cannot be placed by hand |
| **＋ Point** (LAYERS) | a point with no position yet, in the selected animal (else in Scene), selected — and in every camera's list |
| **＋ Animal** (LAYERS) | a new animal with no silhouette yet, selected; give it points (**N** + click, or drag points onto it) and, if you like, outline it with **S** |
| drag points in LAYERS | onto another animal (or one of its points), or onto Scene: they move there and are renamed "*animal part*" in every camera (one Ctrl+Z step) |
| **Delete** (LAYERS, or the key with no timeline window selected) | delete what is selected in LAYERS; for an animal it asks whether its points move to Scene (Yes) or go too (No) |
| **A** | with a calibration: place the selected point at the ◇, where two or more other cameras put it |
| **Alt+click** | with a calibration: where this spot can be in the other cameras (nothing is edited) |
| **Ctrl+click** | move the selected point here |
| **Shift+<** / **Shift+>** | first / last frame the selected point has data on (nothing selected: the silhouette's) |
| **,** / **.** | previous / next frame placed by hand for the selected point |
| right-click a marker (a short click) | clear that point on this frame only, and select it (one Ctrl+Z step). On one of several selected markers: all the selected **points** on this frame, never the silhouette. Does nothing while the Pan tool (**H**) is on |
| hold the right button on a marker (half a second), or right-click its name in LAYERS | its menu: go to its first / last / hand-placed / doubtful frames; clear it on this frame, in the selected window, or entirely; fill its gaps between hand placements with a curve; with a calibration, snap it to the other cameras' rays, or place it at the ◇ |
| (the same menu) | also: rename, delete, lock it to its look, change how it is found (appearance or silhouette), hidden on this frame, may leave its silhouette, **Move to ▸** another animal or Scene, use it as its animal's head |
| two points of one animal selected → right-click | **Connect them with a bone** (again: **Remove the bone between them**) |
| right-click an animal in LAYERS | its menu: rename, outline it (S), show its silhouette, keep its points on its silhouette, midlines, jump to its first / last silhouette, clear silhouettes, save its points and bones as a skeleton template, forget its bones and head, remove it |
| **J** / **Shift+J** | next / previous low-confidence (red) stretch |
| **Shift+X** | mark the selected point hidden on this frame (again to unmark) |
| **Shift+N** | note on this frame |
| Point **▾** | region shape: circle, rectangle, polygon (click the corners, then Enter or a double-click); **Ball marker** — click a ball, SAM outlines it every frame and the fitted circle's centre is the point (balls far apart get a window each, so any spacing works) |
| **S** (the Segment button) | the outlining tool — it outlines the animal selected in LAYERS (the first click makes one when there is none). Click the animal (Shift+click = not the animal, drag = a box around it, right-click a click to remove it); **S** or **Esc** when done |
| **Esc** | cancel whatever you just started: a drag or polygon, the segment tool, the armed crosshair, the pan tool, a half-marked event, a selected stretch of the timeline, the look-here line of an Alt+click — and, with nothing left to cancel, deselect the point |

### Tracking

| Key | Does |
|---|---|
| **Ctrl+A** | select everything to track: every animal and every point (Edit → Select Everything to Track) |
| **Track ▶** or **T** | start tracking from this frame: **only what is selected** in LAYERS — a point row is that point, an animal row its silhouette and all its points (with Track ▾ → **Every camera**: in each camera that has them). A point alone does not bring its animal's silhouette unless the animal keeps its points on its silhouette. Nothing selected = nothing tracked |
| **Shift+T** | several cameras: this run in every camera that has the points here, all at the same time |
| **X** or **Space** | stop (during **3D → Re-track Disagreeing Stretches** it stops the whole queue and asks whether to keep what was re-tracked; during an every-camera run it stops every camera at once) |
| **F** (semi-automatic mode) | track exactly one frame (in every camera with Track ▾ → Every camera) |
| **Ctrl+Z** | undo the last run, bulk edit or hand edit — a click, drag, Ctrl+click, deleted point or Shift+X (one step only) |

### Looking

| Key | Does |
|---|---|
| wheel, **+** / **−**, or the zoom buttons in the picture's lower-right corner | zoom the video (a sideways swipe on a touchpad does not zoom) |
| **R**, or the fit button in that corner | fit the whole picture back in the window |
| **H** (the Pan button) | the tool for looking only: while it is on, a click or a right click on a marker edits nothing. Press **H** again or **Esc** to stop — picking **Point** or **Segment** also stops it. Does nothing until a video is open |
| middle-drag | pan, at any time — even while tracking |
| **Shift + +** / **Shift + −**, **Ctrl+wheel** over the timeline, or the three magnifier buttons in the timeline's top-left corner | zoom the timeline's time axis (the last button shows the whole video) |
| **Ctrl+1** | show / hide the right panel |
| **Ctrl+2** | show only the camera you are working on |
| **Ctrl+Shift+2** | Active view only: only the working camera reads its video; the others stay where they are until Sync all views (again: back to Sync all views; also *View → Other cameras*) |
| **Ctrl+3** | reconstruct the 3D landmarks (needs a calibration of every camera) and show the 3D view |
| **Ctrl+4** | carve the animal's volume at this frame from the cameras' outlines |
| **3D → Set World Axes…** | choose the 3D world's origin, +X and +Y from three landmarks of the 3D result (+Z by the right-hand rule) |
| **Ctrl+5** | show / hide the 3D view (drag to turn it, wheel to zoom, double-click to reset) |
| **Ctrl+6** | show / hide the body side-by-side view (drag the pose to turn it round) |
| **Shift+C** | fix a stale or garbled picture |

### Files

| Key | Does |
|---|---|
| **Ctrl+O** / **Ctrl+Shift+O** | open a video / a project |
| **File → Open Folder of Videos…** | several cameras in one folder: tick which to import, put them in camera order (camera 1 = the reference) |
| **Ctrl+S** / **Ctrl+Shift+S** | save the project / save it under a new name (Ctrl+S during a run: saved as soon as the run stops) |
| **Ctrl+Q** | quit (File → Quit); asks first when there are unsaved changes |
| **Ctrl+E** | export your results |
| **Ctrl+,** | settings (also the last entry of the **Segment ▾** dropdown) |
| **F1** | this manual (**Help → User Manual**) |
| **Help → Check for Updates…** | install a newer version, keeping your projects and models ([section 3](#getting-a-newer-version)) |

### Editing on the timeline

| Key | Does |
|---|---|
| **Shift+drag** | select frames *and* rows |
| **Delete** | clear what the selection covers — or, with no frame window selected, delete what is selected in LAYERS |
| **E**, then **E** | mark the start and end of an event |

---

*A condensed feature summary, the install details and the licences of
everything the program uses are in [../README.md](../README.md).*
