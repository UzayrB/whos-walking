"""python -m openday {live, replay, eval, keytest, bench-checklist, bench-check, screens}. Run from
OpenDay/ with PYTHONDONTWRITEBYTECODE=1 (openday's __init__ also switches bytecode off before anything else)."""

import argparse
import sys

import openday


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m openday", description="Who's walking? Open Day Wi-Fi demo")
    sub = ap.add_subparsers(dest="mode", required=True)
    lv = sub.add_parser("live", help="the Open Day mode, on the boards")
    lv.add_argument("--port", default=None, help="receiver port, e.g. /dev/cu.usbserial-0001")
    lv.add_argument("--reset", action="store_true", help="reset the receiver at start-up (off by default)")
    lv.add_argument("--record", default=None, metavar="DIR", help="BENCH ONLY: write raw CSI and events.jsonl under DIR")
    lv.add_argument("--windowed", action="store_true", help="start in a window instead of full screen")
    lv.add_argument("--new-room", action="store_true",
                    help="the boards were moved: forget today's room level (then Measure room)")
    rp = sub.add_parser("replay", help="the full app fed by thesis recordings instead of the boards (needs ../Code)")
    rp.add_argument("--script", required=True)
    rp.add_argument("--speed", type=float, default=1.0)
    rp.add_argument("--headless", action="store_true", help="no window (tests); the screen still renders off-screen")
    rp.add_argument("--fullscreen", action="store_true")
    ev = sub.add_parser("eval", help="score a home-test (bench) recording: eval --bench DIR")
    ev.add_argument("--bench", required=True, metavar="DIR", help="the folder live --record wrote")
    sub.add_parser("keytest", help="open the window and print every key name received")
    sub.add_parser("bench-checklist", help="print the two-person bench session protocol")
    bk = sub.add_parser("bench-check", help="walk detection on recorded runs (prints only, writes nothing)")
    bk.add_argument("dir", metavar="DIR", help="a run folder (with bench_RX.bin) or a folder of runs")
    bk.add_argument("--max-s", type=float, default=None, help="override walk.max_s")
    sc = sub.add_parser("screens", help="a picture of every screen into reports/screens/ (no boards needed; "
                                        "replaces its own old pictures there, never other files)")
    sc.add_argument("--out", default=None, metavar="DIR", help="another folder instead of reports/screens/")
    a = ap.parse_args(argv)
    if a.mode == "replay" and a.speed < 1.0:
        ap.error("--speed must be at least 1.0 (the click-to-receiver-clock mapping assumes real time or faster)")
    cfg = openday.load_config()

    if a.mode == "bench-checklist":
        import time
        from openday import bench
        print(bench.BENCH_CHECKLIST.format(day=time.strftime("%Y%m%d")))
        return 0
    if a.mode == "screens":
        from openday import screenshots
        return screenshots.main(cfg, a.out)
    if a.mode == "bench-check":
        from openday import bench
        return bench.bench_check(cfg, a.dir, a.max_s)
    if a.mode == "eval":
        openday.use_code()   # stop here, before any progress line, on a copy without the thesis code
        from openday import bench
        bench.run_bench(cfg, a.bench)
        return 0
    from openday import app
    if a.mode == "keytest":
        return app.keytest(cfg)
    if a.mode == "replay":
        return app.run_replay(cfg, a.script, speed=a.speed, headless=a.headless, fullscreen=a.fullscreen)
    return app.run_live(cfg, port=a.port, reset=a.reset, record=a.record, fullscreen=not a.windowed,
                        new_room=a.new_room)


if __name__ == "__main__":
    sys.exit(main())
