"""Run the agents without opening the app — for a schedule (cron / Task Scheduler).

  python run_agents.py daily        # early-warning drafts + data checks         (every school morning)
  python run_agents.py weekly       # Principal briefing                         (Monday 7:30)
  python run_agents.py after-exam   # action review + remedial plans + re-check  (after marks are entered)
  python run_agents.py all

  options: --data DIR   data folder (default ./data or $SCHOOL_DATA_DIR)      --no-llm   never call Claude

Example crontab:
  30 7 * * 1   cd /path/to/school_analytics && python run_agents.py weekly
  0  8 * * 1-6 cd /path/to/school_analytics && python run_agents.py daily

Nothing is sent and no marks are changed: drafts wait in data/agent_queue.csv until a person approves them in the app.
Outputs are saved in data/agent_outputs/.
"""
import argparse, json, os, sys
import agents as G


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", choices=["daily", "weekly", "after-exam", "all"])
    ap.add_argument("--data"); ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args(argv)
    d = G.SchoolData(a.data)
    which = ("daily", "after-exam", "weekly") if a.job == "all" else (a.job,)
    out = G.run_all(d, which, use_llm=not a.no_llm)
    od = d.path("agent_outputs"); os.makedirs(od, exist_ok=True)
    day = d.today.isoformat()
    if "briefing" in out:
        open(os.path.join(od, f"briefing_{day}.md"), "w", encoding="utf-8").write(G.briefing_markdown(out["briefing"]))
        json.dump(out["briefing"], open(os.path.join(od, f"briefing_{day}.json"), "w", encoding="utf-8"), indent=1, default=str, ensure_ascii=False)
        print(G.briefing_markdown(out["briefing"]))
    if "plan_recheck" in out and out["plan_recheck"]:
        json.dump(out["plan_recheck"], open(os.path.join(od, f"plan_recheck_{day}.json"), "w", encoding="utf-8"), indent=1, default=str, ensure_ascii=False)
    new = {k: v["items_created"] for k, v in out.items() if isinstance(v, dict) and "items_created" in v}
    print("Drafts added to the approval inbox:", new or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
