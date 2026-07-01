"""Preview and save an Autopilot plan."""

from easylora import autopilot_plan, save_autopilot_report


def main() -> None:
    plan = autopilot_plan(
        model="meta-llama/Llama-3.2-1B",
        dataset="tatsu-lab/alpaca",
        quality="balanced",
    )
    print(plan.to_markdown())
    path = save_autopilot_report(plan, "./output")
    print(f"Report written to: {path}")


if __name__ == "__main__":
    main()
