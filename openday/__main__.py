"""python -m openday {live, replay, eval, eval-pair, keytest, bench-checklist, bench-check, screens}. Run from
OpenDay/ with PYTHONDONTWRITEBYTECODE=1 (openday's __init__ also switches bytecode off before anything else)."""

import argparse
import sys

import openday


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m openday", description="Who's walking? Open Day Wi-Fi demo")
    sub = ap.add_subparsers(dest="mode", required=True)
    lv = sub.add_parser("live", help="the Open Day mode, on the boards")
    lv.add_argument("--port", default=None, help="receiver port, e.g. /dev/cu.usbserial-0001")
    lv.add_argument("--identify", action="store_true",
                    help="several ports, none streaming: find the receiver with the thesis board identifier (pulses each board's reset once)")
    lv.add_argument("--reset", action="store_true", help="reset the receiver at start-up (off by default)")
    lv.add_argument("--record", default=None, metavar="DIR", help="BENCH ONLY: write raw CSI and events.jsonl under DIR")
    lv.add_argument("--windowed", action="store_true", help="start in a window instead of full screen")
    rp = sub.add_parser("replay", help="the full app fed by thesis recordings instead of the boards (needs ../Code)")
    rp.add_argument("--script", required=True)
    rp.add_argument("--speed", type=float, default=1.0)
    rp.add_argument("--headless", action="store_true", help="no window (tests); the screen still renders off-screen")
    rp.add_argument("--fullscreen", action="store_true")
    ev = sub.add_parser("eval", help="offline evaluation on the thesis recordings, or score a bench session")
    ev.add_argument("--bench", default=None, metavar="DIR")
    ev.add_argument("--no-write", action="store_true", help="do not write the fitted values into the config")
    ep = sub.add_parser("eval-pair", help="leave two people out: compare the decision models and print the report")
    ep.add_argument("--draws", type=int, default=50)
    ep.add_argument("--refresh", action="store_true", help="re-run every walk through the chain (else the cache)")
    ep.add_argument("--write", action=argparse.BooleanOptionalAction, default=False,
                    help="also replace the shipped model (state/model) and its config lines, and save the report in "
                         "reports/ (default: --no-write, print the report only)")
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
    if a.mode in ("eval-pair", "eval", "bench-check"):
        openday.use_code()   # stop here, before any progress line, on a copy without the thesis code
    if a.mode == "eval-pair":
        from openday import evaluate
        if a.refresh:
            evaluate.load_rows(cfg, refresh=True)
        text = evaluate.run_pair(cfg, draws=a.draws, write=a.write)
        if not a.write:
            print(text)
        return 0
    if a.mode == "eval":
        if a.bench:
            from openday import bench
            bench.run_bench(cfg, a.bench)
        else:
            from openday import evaluate
            evaluate.run(cfg, write=not a.no_write)
        return 0
    from openday import app
    if a.mode == "keytest":
        return app.keytest(cfg)
    if a.mode == "replay":
        return app.run_replay(cfg, a.script, speed=a.speed, headless=a.headless, fullscreen=a.fullscreen)
    return app.run_live(cfg, port=a.port, identify=a.identify, reset=a.reset, record=a.record,
                        fullscreen=not a.windowed)


if __name__ == "__main__":
    sys.exit(main())
