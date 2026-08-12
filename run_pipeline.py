"""
run_pipeline.py
Runs the full daily pipeline in order:
  1. fetch_jobs.py              (pull latest postings, accumulate into raw CSV)
  2. clean_data.py              (dedupe, fix salary values)
  3. clean_and_extract.py       (skill + experience-level extraction)
  4. fetch_full_descriptions.py (gap-fill skills for zero-skill rows)
  5. eda.py                     (regenerate charts)

Meant to be triggered automatically (e.g. by Windows Task Scheduler) once
a day. Every run's output is appended to pipeline_log.txt with a
timestamp, so you can check later whether a scheduled run succeeded
without having to be at the machine when it runs.

Safe to also run manually any time: python run_pipeline.py
"""

import subprocess
import sys
from datetime import datetime

STEPS = [
    "fetch_jobs.py",
    "clean_data.py",
    "clean_and_extract.py",
    "fetch_full_descriptions.py",
    "eda.py",
]

LOG_FILE = "pipeline_log.txt"


def run_step(script, log):
    header = f"\n{'=' * 60}\n{script}  —  {datetime.now().isoformat(timespec='seconds')}\n{'=' * 60}\n"
    print(header, end="")
    log.write(header)
    log.flush()

    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    print(output)
    log.write(output)
    log.flush()

    if result.returncode != 0:
        msg = f"\n!!! {script} FAILED (exit code {result.returncode}) — stopping pipeline.\n"
        print(msg)
        log.write(msg)
        return False
    return True


def main():
    start = datetime.now()
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n\n########## PIPELINE RUN START: {start.isoformat(timespec='seconds')} ##########\n")
        for script in STEPS:
            ok = run_step(script, log)
            if not ok:
                log.write(f"########## PIPELINE RUN ABORTED: {datetime.now().isoformat(timespec='seconds')} ##########\n")
                sys.exit(1)
        end = datetime.now()
        summary = f"\n########## PIPELINE RUN COMPLETE: {end.isoformat(timespec='seconds')} (took {end - start}) ##########\n"
        print(summary)
        log.write(summary)


if __name__ == "__main__":
    main()
