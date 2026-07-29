# Contributing to fftkit

Contributions are genuinely welcome, and that includes the ones that are not
code. A bug report, a confusing docstring, a README paragraph that turned out to
be wrong, a question that took you an hour to answer yourself — all of those are
worth opening an [issue](https://github.com/openfluids/fftkit/issues) for.

If you are unsure whether something is worth reporting, it probably is. Open the
issue.

## Getting set up

```bash
git clone https://github.com/openfluids/fftkit.git
cd fftkit
uv sync
```

That gives you the core package and the test tooling. The optional FFT backends
(`pyfftw`, `mkl`, `torch`, `tensorflow`, CUDA) are not needed to work on
fftkit — every backend is probed at import time and reports itself unavailable
rather than failing, so the test suite passes without any of them. If you are
working on backend code specifically:

```bash
uv sync --group backends
```

## Before you open a pull request

The same three checks CI runs:

```bash
uv run pytest -v
uv run ruff check .
uv run mypy --strict src/fftkit/ --config-file mypy.ini
```

If one fails for a reason you think is unrelated to your change, say so in the
pull request rather than working around it — that is useful information, and
sometimes it is CI that is wrong.

CI additionally tests against Python 3.11 through 3.14. You do not need to run
all four locally; the matrix will tell you.

## What makes a pull request easy to review

- **One thing at a time.** A focused change gets reviewed quickly. A change that
  also reformats fifty unrelated lines is hard to read and slow to merge.
- **Say what you verified.** A pasted command and its output is worth more than
  "tested locally".
- **Ask early.** For anything substantial, open an issue first. It is much
  better to disagree about an approach before you have written it than after.
- **Draft PRs are fine.** Opening one early to ask "is this the right
  direction?" is welcome and costs nothing.

Reviews may take a few days — one maintainer, research alongside. A nudge on a
quiet pull request is welcome, not annoying.

## Conventions

Only the ones that are actually enforced:

- Type annotations on public API; `mypy --strict` must pass for `src/fftkit/`.
- Formatting and import order are handled by `ruff` — do not hand-tune them.
- New backends must degrade gracefully: probe at import, report unavailable,
  never raise on a missing optional dependency.
- Use `np.random.default_rng(seed)`, not the legacy module-level `np.random`
  calls.

## Conduct and licence

Everyone taking part is asked to follow the
[openfluids Code of Conduct](https://github.com/openfluids/.github/blob/main/CODE_OF_CONDUCT.md).
It is short.

fftkit is licensed under Apache-2.0, and contributions are accepted under the
same licence. See `LICENSE` and `NOTICE`.

Found a security problem? Please do not open a public issue — see the
[security policy](https://github.com/openfluids/fftkit/security/policy).
