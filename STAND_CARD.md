# Stand card

The routine for Open Day, in short. The tally sheet is at the end: print it as many times as you need.
To print: open this file where it shows as a page (for example on GitHub, in Safari), then File > Print.
Details: README, section 3, "Your routine at the stand".

## Before visitors arrive

- **Details**: about 150 pkt/s on the bottom line. Click it again.
- The light goes red under 100 pkt/s.
- **Measure room** while the hall is still empty. The app keeps this room level all day.
- Tape the crowd line 3 m from the boards.

## The room level

- Don't measure again unless it is quiet: nobody within 3 m moving.
- Only a clearly quieter measurement replaces today's level ("Quieter than earlier"). Then your kept walks expire.
- "Busier than earlier", "About the same as earlier" or "Someone moved: kept the earlier room level ...":
  nothing to do. Your walks are kept.
- After a restart: **Use last measurement**.
- Moved the boards? **Quit**, **Yes, quit**, then `./run_openday.command --new-room`, then **Measure room**
  (README, section 6).
- Light stays amber ("Wait for green") with nobody moving anywhere, even after Measure room? Something near the
  boards changed: **Quit**, **Yes, quit**, `./run_openday.command --new-room`, then **Measure room**.

## Your walks

- One warm-up walk first: **Start walk** on "Who's walking?" with no player. It counts for nothing.
- Then **Redo my walks**: your five.

## Every walk

- Say "go" and click as you say it.
- The walker leaves on "go". Yours too.
- People moving nearby? Click **They've stopped** as each walker lands on the X.
- Light says "Wait for green", but nobody inside the 3 m line? Start the walk anyway.

## Say it

To every walker:

> "Stand on START. When I say go, walk to the X and stand still on it until the answer appears. Then walk
> back."

## Keep score

- Mark every answer on the tally sheet. The screen keeps no score.

<div style="page-break-after: always;"></div>

# Tally sheet

Date: ____________ Sheet ____ of ____

V = the visitor. U = you. One row per answer. "Walk once more!" and its answer are one row: tick "Walked twice?".

| # | Who walked (V/U) | Right? (tick or cross) | Walked twice? | Notes |
|---:|:---:|:---:|:---:|:---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| 6 | | | | |
| 7 | | | | |
| 8 | | | | |
| 9 | | | | |
| 10 | | | | |
| 11 | | | | |
| 12 | | | | |
| 13 | | | | |
| 14 | | | | |
| 15 | | | | |
| 16 | | | | |
| 17 | | | | |
| 18 | | | | |
| 19 | | | | |
| 20 | | | | |

**Totals**

- Answers: ______ Right: ______ Wrong: ______ Walked twice: ______
- You walked: right ______ of ______
- The visitor walked: right ______ of ______
