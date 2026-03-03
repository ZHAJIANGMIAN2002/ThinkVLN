# Next Week TODO (2026-03-01 to 2026-03-07)

Main objective: lock a reliable Actor-to-Watcher pipeline so results are measurable and Watcher training is unblocked.

- Build subtask-level evaluation and report `action_acc`, `progress_mae`, and closed-loop `SR/SPL`.
- Run three Actor variants (`parallel`, `AR`, `AR+binning`) and record quality-speed tradeoffs in one table.
- Generate first failure dataset with Oracle Judge labels (`FAIL/RESUME/PROCEED`) and class distribution stats.
- Train Watcher v0 with explicit CoT decisions only (`resume/proceed/fail`), without GRPO this week.
- Add quick debug visualization (GT vs Actor trajectory + Watcher decision points) for failed episodes.


