"""
run_pipeline.py
Runs the full daily pipeline in order:
  1. fetch_jobs.py              (pull latest postings, accumulate into raw CSV)
  2. clean_data.py              (dedupe, fix salary values)
  3. clean_and_extract.py       (skill + experience-level extraction)
  4. fetch_full_descriptions.py (gap-fill skills for zero-skill rows)
  5. eda.py                     (regenerate charts)
  6. build_vector_store.py      (rebuild the RAG vector store with fresh data)
  7. git add + commit + push    (so the deployed Streamlit Cloud app also
                                  picks up today's data — without this, the
                                  live app keeps showing whatever was last
                                  pushed, even though local data is fresh)

Meant to be triggered automatically (e.g. by Windows Task Scheduler) once
a day. Every run's output is appended to pipeline_log.txt with a
timestamp, so you can check later whether a scheduled run succeeded
without having to be at the machine when it runs.

Safe to also run manually any time: python run_pipeline.py

NOTE: the git push step assumes you've already run `git push` successfully
once by hand (so credentials are cached / SSH is set up) — it just repeats
the same auth your terminal already knows.
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
    "build_vector_store.py",
]

# Files/folders whose daily changes should be pushed so the live app updates.
# chroma_db is intentionally excluded from the *pipeline's* auto-push — its
# binary files change completely every rebuild and would bloat the repo's
# history if committed daily. See the README note on this tradeoff.
GIT_PATHS = [
    "da_job_postings_raw.csv",
    "da_job_postings_clean.csv",
    "charts",
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


def push_to_github(log):
    header = f"\n{'=' * 60}\ngit commit + push  —  {datetime.now().isoformat(timespec='seconds')}\n{'=' * 60}\n"
    print(header, end="")
    log.write(header)
    log.flush()

    commands = [
        ["git", "add"] + GIT_PATHS,
        ["git", "commit", "-m", f"Daily data update {datetime.now().strftime('%Y-%m-%d')}"],
        ["git", "push"],
    ]
    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = f"$ {' '.join(cmd)}\n{result.stdout}{result.stderr}\n"
        print(output)
        log.write(output)
        log.flush()
        # "nothing to commit" isn't a real failure — just means no data
        # changed since yesterday's run, which is fine, skip straight to push
        if result.returncode != 0 and "nothing to commit" not in output:
            if cmd[1] == "commit":
                print("Nothing new to commit — skipping push.\n")
                log.write("Nothing new to commit — skipping push.\n")
                return True
            msg = f"\n!!! git step failed (exit code {result.returncode}) — check GitHub auth.\n"
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

        push_to_github(log)

        end = datetime.now()
        summary = f"\n########## PIPELINE RUN COMPLETE: {end.isoformat(timespec='seconds')} (took {end - start}) ##########\n"
        print(summary)
        log.write(summary)


if __name__ == "__main__":
    main()
