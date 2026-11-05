# darvin-auto

A lightweight agent workflow and skills for investigating a Darvin leaderboard,
finding a baseline, preparing an inference image and tracking an authorized
submission. The offline decision planner and tests are experimental.

This is not a turnkey universal executor: a fresh URL-to-score run has not been
verified as fully automatic, and authentication, permissions, incomplete
contracts and resource limits may require intervention. This release contains
no live platform executor, baseline model, weights, private requirements or
submission credentials.

## Layout and prerequisites

- `workflows/run-leaderboard.md`: entry workflow.
- `skills/darvin-auto-benchmark`: orchestration skill and offline planner.
- `skills/open-source-baseline`: source selection and handoff skill.
- `skills/darvin-image`: image build, test and publication skill.

Install applicable Darvin and Feishu CLI/tools and skills separately,
authenticate them, and consult their current official documentation. Provide a
leaderboard URL and authorized resource/network limits to an agent with these
skills available. Verify the real task contract and reconcile uncertain writes
before retrying.

## Tests

```sh
(cd skills/darvin-auto-benchmark && PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests)
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s skills/open-source-baseline/scripts -p 'test*.py'
```

The baseline tests require Pillow. Tests cover offline components only, not a live submission. The optional image
authentication handoff smoke test requires a separately supplied adapter and
dependencies. No license has been specified for this repository; separately
acquired code and weights have their own licenses.
