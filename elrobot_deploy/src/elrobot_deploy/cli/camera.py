"""`elrobot_deploy camera ...`: one camera on the bench, no robot and no session.

    elrobot_deploy camera list                  what AVFoundation, USB and the LAN (camera
                                                phones) see
    elrobot_deploy camera probe [--camera N]    its UVC controls, before and while streaming
                                                (can exposure be set on this Mac?)
    elrobot_deploy camera test [--camera N] [--seconds S] [--set ...]
                                                run it as a session would (P2's own code:
                                                open, fit the lens, the exposure mode) and say,
                                                once a second, what the exposure is doing;
                                                saves the full frame (lens circle drawn), the
                                                square, and a CSV of every metered frame

`test` answers "are the camera's own settings good enough?": with the default `camera_first`
it says whether, and why, it took over; `--set cameras.wrist.exposure="camera"` keeps the
camera's own settings throughout and reports how clipped the disc is.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from elrobot_deploy.config_loading.config import REPO, load
from elrobot_deploy.config_loading.errors import ConfigError


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="elrobot_deploy camera",
                                description="One camera on the bench (no robot).")
    sub = p.add_subparsers(dest="what", required=True, metavar="list|probe|test")
    sub.add_parser("list", help="the cameras AVFoundation and USB see")
    for name, text in (("probe", "the camera's UVC controls, before and while streaming"),
                       ("test", "run it as a session would, and report")):
        s = sub.add_parser(name, help=text)
        s.add_argument("--profile", default=None)
        s.add_argument("--set", action="append", default=[], dest="sets",
                       metavar="SECTION.KEY=VALUE")
        s.add_argument("--camera", default=None, help="which [cameras.<name>] (default: the "
                                                      "first)")
    t = sub.choices["test"]
    t.add_argument("--seconds", type=float, default=20.0)
    t.add_argument("--out", default=str(REPO / "logs" / "camera"),
                   help="where the snapshots and the CSV go (default: ElRobot/logs/camera)")
    return p


def _camera(args):
    cfg = load(args.profile, sets=args.sets)
    if not cfg.cameras:
        raise ConfigError("the profile declares no [cameras.<name>]")
    return cfg.camera(args.camera) if args.camera else cfg.cameras[0]


def do_list() -> int:
    from elrobot_deploy.runtime.cameras import sources, uvc

    print("AVFoundation (what can stream):")
    try:
        import AVFoundation as AVF
        for d in sources.avf_devices():
            usb = d["usb"]
            where = (f"USB {usb[1]:04x}:{usb[2]:04x} @ {usb[0]:#010x}" if usb
                     else "not USB (built-in / Continuity)")
            print(f"  {d['name']!r}  {d['uid']}  [{where}]")
            if usb:
                dev = AVF.AVCaptureDevice.deviceWithUniqueID_(d["uid"])
                modes = sorted({(f["width"], f["height"], f["fourcc"],
                                 max(hi for _, hi in f["rates"]))
                                for f in sources.avf_formats(dev)}, reverse=True)
                for w, h, cc, fps in modes:
                    print(f"      {w}x{h} {cc} up to {fps:.0f} fps")
    except ImportError as exc:
        print(f"  (AVFoundation unavailable: {exc}; run `uv sync`)")
    print("Camera phones on the LAN (the app's Camera role, source = \"phone\"):")
    try:
        from elrobot_deploy.runtime.cameras.phone_camera import browse
        phones = browse()
        for d in phones:
            print(f"  {d['name']!r}  {d['host']}:{d['port']}")
        if not phones:
            print("  none (the app open in the Camera role, same WiFi, Local Network allowed?)")
    except Exception as exc:  # noqa: BLE001
        print(f"  ({type(exc).__name__}: {exc})")
    print("USB (what has UVC controls):")
    try:
        found = uvc.list_devices()
        for d in found:
            print(f"  {d.line()}")
        if not found:
            print("  none. A camera that is plugged in but not listed here (nor by "
                  "`system_profiler SPUSBDataType`) is not enumerating: cable, connector, port.")
    except Exception as exc:  # noqa: BLE001
        print(f"  ({type(exc).__name__}: {exc})")
    return 0


def _print_probe(ctl, title: str) -> dict:
    got = ctl.probe()
    print(title)
    for name, r in got.items():
        if "error" in r:
            print(f"  {name:<12} {r['error']}")
        else:
            cur = r.get("cur", f"(read failed: {r.get('cur_error')})")
            print(f"  {name:<12} cur {cur!s:>6}   min {r['min']:>6}  max {r['max']:>6}  "
                  f"res {r['res']:>5}  def {r['def']:>6}")
    return got


def do_probe(args) -> int:
    from elrobot_deploy.runtime.cameras import make_source, sources
    from elrobot_deploy.runtime.cameras.uvc import UvcControls

    cam = _camera(args)
    print(f"camera {cam.name}: {cam.line()}")
    src = make_source(cam)
    # 1. The control channel with nothing streaming (needs no camera permission).
    try:
        if isinstance(src, sources.AvfCamera):
            info = sources.resolve_avf(cam.device)
            usb = sources.parse_usb_uid(info["uid"])
            if usb is None:
                raise sources.CameraError(f"{info['name']!r} is not a USB camera")
            print(f"device: {info['name']!r} {info['uid']} (USB {usb[1]:04x}:{usb[2]:04x} @ "
                  f"{usb[0]:#010x})")
            ctl = UvcControls.open(vid=usb[1], pid=usb[2], location=usb[0])
        else:
            src.connect()
            ctl = src.controls()
    except Exception as exc:  # noqa: BLE001
        print(f"NO CONTROL CHANNEL: {type(exc).__name__}: {exc}")
        print("=> exposure can only be the camera's own here (camera_first stays on it).")
        return 1
    if ctl is None:
        print("this kind of camera has no control channel in this package")
        return 1
    got = _print_probe(ctl, "controls, not streaming:")
    ae = got.get("ae_mode", {})
    if "res" in ae:
        modes = [n for bit, n in ((1, "manual"), (2, "auto"), (4, "shutter priority"),
                                  (8, "aperture priority")) if ae["res"] & bit]
        print(f"auto-exposure modes it offers: {', '.join(modes)}")

    # 2. The same while frames flow, which is the case that matters: does a write change
    #    the image? Manual exposure at a quarter of the current one must darken it.
    try:
        if isinstance(src, sources.AvfCamera):
            src.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"cannot stream: {exc}")
        ctl.close()
        return 1
    print(f"streaming: {src.describe()}")
    px = np.empty(cam.shape, np.uint8)

    def mean_after(n=12):
        for _ in range(n):
            src.read_into(px)
        return float(px.mean())

    ok = False
    try:
        before = mean_after()
        _print_probe(ctl, "controls, while streaming:")
        mode, exp = ctl.get("ae_mode"), ctl.get("exposure")
        ctl.set("ae_mode", 1)
        low = max(ctl.range("exposure").get("min", 1), exp // 4)
        ctl.set("exposure", low)
        darker = mean_after()
        ctl.write("exposure", None)         # while still manual: auto refuses the write
        ctl.set("ae_mode", mode)
        after = mean_after(30)
        print(f"frame mean: {before:.1f} as found; {darker:.1f} with manual exposure "
              f"{low} (was {exp}); {after:.1f} after handing back (ae_mode {ctl.get('ae_mode')})")
        ok = darker < 0.8 * before
        if not ok:
            print("the write was accepted but the image did not darken")
    except Exception as exc:  # noqa: BLE001
        print(f"WRITE FAILED while streaming: {exc}")
    print("=> " + ("exposure CAN be set while streaming on this Mac: every exposure mode works."
                   if ok else "exposure cannot be set while streaming: camera_first stays on "
                              "the camera's own settings."))
    ctl.close()
    src.close()
    return 0 if ok else 1


def do_test(args) -> int:
    from elrobot_deploy.runtime.processes.p2_camera import CameraRunner

    cam = _camera(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = out / f"{stamp}_{cam.name}"
    print(f"camera {cam.name}: {cam.line()}")
    runner = CameraRunner(cam)
    px = np.empty(cam.shape, np.uint8)
    rows = []
    try:
        runner.start(px)
        _snap(px, runner, f"{base}_start")
        t_end = time.monotonic() + args.seconds
        t_print = 0.0
        while time.monotonic() < t_end:
            if not runner.frame(px):
                print("the stream stopped")
                break
            c = runner.controller
            if runner.frames % 3 == 0 and c.last is not None:
                _, exp, gain = runner.exposure_in_effect()
                rows.append([round(runner.source.t_capture, 4), c.owner, exp, gain,
                             round(c.last.p995, 1), round(c.last.mean, 1),
                             round(c.last.clip_frac * 100, 3), c.spell.need or ""])
            if time.monotonic() >= t_print:
                t_print = time.monotonic() + 1.0
                s = runner.status()
                ex = s["exposure"]
                own = (f"ours exp={ex['exposure']} gain={ex['gain']} ({ex['phase']})"
                       if ex["owner"] == "ours" else
                       f"camera's own (reads exp={s['readback'][0]} gain={s['readback'][1]})")
                print(f"  {s['fps']:5.1f} fps  p99.5 {ex['p995']}  mean {ex['mean']}  clipped "
                      f"{ex['clip_pct']}%  {own}"
                      + (f"  need {ex['need']} {ex['need_s']}s" if ex["need"] else ""))
        _snap(px, runner, f"{base}_end")
    finally:
        runner.close()
    with open(f"{base}_exposure.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "owner", "exposure", "gain", "p995", "mean", "clip_pct", "need"])
        w.writerows(rows)
    s = runner.status()
    print(json.dumps({k: s[k] for k in ("device", "fps", "dropped", "lens_line", "controls",
                                        "exposure")}, indent=1))
    clipped = [r[6] for r in rows if r[1] == "camera"]
    if clipped:
        print(f"while the camera's own exposure ran: disc clipped (>=250) "
              f"median {np.median(clipped):.2f}%, max {max(clipped):.2f}% "
              f"over {len(clipped)} metered frames (g1d measured 26% on its rig).")
    ex = s["exposure"]
    if ex["takeover"]:
        print(f"camera_first TOOK OVER: {ex['takeover']}")
    elif ex["mode"] == "camera_first":
        print("camera_first did not need to take over: the camera's own settings were enough "
              "for this scene.")
    print(f"saved {base}_start_full.jpg, _start_square.jpg, _end_*.jpg, _exposure.csv")
    return 0


def _snap(px, runner, prefix: str) -> None:
    import cv2

    from elrobot_deploy.runtime.cameras import lens

    full = px.copy()
    d = runner.decision
    if d is not None:
        cx, cy, r = d.lens_circle
        cv2.circle(full, (int(round(cx)), int(round(cy))), int(round(r)), (0, 255, 0), 2)
        x0, y0, x1, y1 = runner.crop
        cv2.rectangle(full, (x0, y0), (x1 - 1, y1 - 1), (255, 128, 0), 1)
    cv2.imwrite(f"{prefix}_full.jpg", full)
    x0, y0, x1, y1 = runner.crop
    if x1 - x0 == y1 - y0:
        cv2.imwrite(f"{prefix}_square.jpg", lens.cut(px, x0, y0, x1 - x0))


def main(argv) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.what == "list":
            return do_list()
        if args.what == "probe":
            return do_probe(args)
        return do_test(args)
    except ConfigError as exc:
        print(f"elrobot_deploy camera: {exc}")
        return 2
    except Exception as exc:
        from elrobot_deploy.runtime.cameras import CameraError
        if isinstance(exc, CameraError):
            print(f"elrobot_deploy camera: {exc}")
            return 1
        raise


__all__ = ["main"]
