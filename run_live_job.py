#!/usr/bin/env python3
"""
RUN THE WHOLE LIVE JOB with one command.
Lokoja LGA flood susceptibility project.

Steps, in order (the job stops at the first one that fails and says which):
  1. fetch_imerg_early.py      observed rain up to the issue date T0 (IMERG Early)
  2. fetch_ecmwf_forecast.py   forecast rain for T0+1 and T0+2 (ECMWF open data)
  3. fetch_smap_latest.py      latest soil moisture (SMAP L4, through Earth Engine)
  4. phase6c_grid_job.py       scores DTI and FRI for now, +24 h and +48 h
  5. phase7a_make_web_data.py  rebuilds the web map tiles and JSON
  6. (optional) copies web_data into the web app's public folder

Everything is logged to logs/live_<T0>.txt as well as printed.

Run     : python run_live_job.py --project YOUR_EE_PROJECT_ID
Options : --dry-run        print the commands without running them
          --skip-fetch     reuse the files already in data/phase6_inputs (re-score and rebuild only)
          --app-public PATH  copy web_data into this folder when the job finishes
                             (for example C:\\Users\\jatto\\lokoja-flood-app\\public\\web_data)

Use the same command window as for the other scripts (the one from flood_shell.bat).
"""
import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime

INPUT_DIR = "data/phase6_inputs"
LOG_DIR = "logs"


def run(cmd, logfile, dry):
    line = "$ " + " ".join(cmd)
    print(line, flush=True)
    logfile.write(line + "\n")
    if dry:
        return 0
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for out in proc.stdout:
        print(out, end="", flush=True)
        logfile.write(out)
    proc.wait()
    logfile.flush()
    return proc.returncode


def read_t0():
    p = os.path.join(INPUT_DIR, "t0.txt")
    if not os.path.exists(p):
        return None
    return open(p).read().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--app-public", default=None)
    args = ap.parse_args()

    py = sys.executable
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(LOG_DIR, "live_%s.txt" % stamp), "w") as logfile:
        logfile.write("Live job started %s\n" % datetime.now().isoformat(timespec="seconds"))

        steps = []
        if not args.skip_fetch:
            if not args.project:
                sys.exit("Give your Earth Engine project with --project (or set EE_PROJECT).")
            steps += [
                ("IMERG observed rain", [py, "fetch_imerg_early.py"]),
                ("ECMWF forecast rain", [py, "fetch_ecmwf_forecast.py"]),
                ("SMAP soil moisture", [py, "fetch_smap_latest.py", "--project", args.project]),
            ]
        for name, cmd in steps:
            print("\n=== %s ===" % name, flush=True)
            if run(cmd, logfile, args.dry_run) != 0:
                sys.exit("\nSTOPPED at '%s'. Read the messages above. Nothing after this step was run." % name)

        t0 = read_t0() if not args.dry_run else "<T0>"
        if not t0:
            sys.exit("No issue date found (%s/t0.txt). The IMERG step must run first." % INPUT_DIR)
        tail = [
            ("Score DTI and FRI", [py, "phase6c_grid_job.py", "--t0", t0]),
            ("Build web tiles", [py, "phase7a_make_web_data.py"]),
        ]
        for name, cmd in tail:
            print("\n=== %s ===" % name, flush=True)
            if run(cmd, logfile, args.dry_run) != 0:
                sys.exit("\nSTOPPED at '%s'. Read the messages above." % name)

        if args.app_public and not args.dry_run:
            print("\n=== Copy web data into the app ===", flush=True)
            shutil.copytree("web_data", args.app_public, dirs_exist_ok=True)
            print("Copied web_data to %s" % args.app_public)
            logfile.write("Copied web_data to %s\n" % args.app_public)
        print("\nLive job finished. Issue date: %s" % t0)
        logfile.write("Finished %s\n" % datetime.now().isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()