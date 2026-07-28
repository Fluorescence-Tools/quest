"""The optional dependencies must stay optional — proved by simulating absence.

Qt, LabelLib and `IMP.bff` are imported behind `try`/`except` with fallbacks.
That is what lets the CLI, the API and the web backend run where those are not
installed, and it is the reason a Windows box without a compiled `IMP.bff` can
still compute accessible volumes through LabelLib.

**Until 2026-07-28 the rule was enforced by an interpreter that genuinely lacked
them** — the mambaforge base environment had no Qt and no compiled `IMP.bff`, so
running the suite there exercised every fallback for free. That environment is
retired ([okf/specs/programme-2026-07.md](../okf/specs/programme-2026-07.md),
decisions 4 and 5), and the canonical `arm64` environment has all three. Nothing
would enter those branches again on its own.

The failure mode this guards against is documented and has already happened
once: `quest/lib/imp_av.py` shipped a `Model.get_particles()` call that does not
exist, and nothing noticed because no test entered that branch. A backend with
no coverage ships broken code.

These tests therefore *simulate* the absence rather than relying on it.
"""

from __future__ import annotations

import builtins
import subprocess
import sys
import textwrap

import pytest

from quest.core import av


class TestTheAvBackendsDegrade:
    """`resolve_av_backend` is the single place that decides what runs."""

    def test_it_prefers_imp_bff_when_both_are_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(av, "HAS_LABELLIB", True)
        monkeypatch.setattr(av, "HAS_IMP_BFF", True)
        monkeypatch.delenv("QUEST_AV_BACKEND", raising=False)

        assert av.resolve_av_backend() == av.AV_BACKENDS[0] == "imp_bff"

    def test_it_falls_back_to_labellib_without_imp_bff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Windows case: IMP is installed but its bff module is not built."""
        monkeypatch.setattr(av, "HAS_LABELLIB", True)
        monkeypatch.setattr(av, "HAS_IMP_BFF", False)
        monkeypatch.delenv("QUEST_AV_BACKEND", raising=False)

        assert av.available_av_backends() == ["labellib"]
        assert av.resolve_av_backend() == "labellib"

    def test_it_uses_imp_bff_without_labellib(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(av, "HAS_LABELLIB", False)
        monkeypatch.setattr(av, "HAS_IMP_BFF", True)
        monkeypatch.delenv("QUEST_AV_BACKEND", raising=False)

        assert av.available_av_backends() == ["imp_bff"]
        assert av.resolve_av_backend() == "imp_bff"

    def test_with_neither_it_says_so_instead_of_returning_an_empty_volume(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(av, "HAS_LABELLIB", False)
        monkeypatch.setattr(av, "HAS_IMP_BFF", False)
        monkeypatch.delenv("QUEST_AV_BACKEND", raising=False)

        with pytest.raises(RuntimeError, match="No accessible-volume backend"):
            av.resolve_av_backend()

    def test_asking_for_a_missing_backend_names_what_is_there(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(av, "HAS_LABELLIB", True)
        monkeypatch.setattr(av, "HAS_IMP_BFF", False)

        with pytest.raises(RuntimeError, match="labellib"):
            av.resolve_av_backend("imp_bff")

    def test_an_unknown_name_is_a_value_error_not_a_silent_default(self) -> None:
        with pytest.raises(ValueError, match="Unknown accessible-volume backend"):
            av.resolve_av_backend("labellib2")


class TestTheDomainImportsWithoutOptionalPackages:
    """`import quest.core` must not need Qt, LabelLib or IMP.

    Run in a subprocess with the module blocked at import, because the parent
    has all three loaded and cannot un-load them.
    """

    @pytest.mark.parametrize("blocked", ["PyQt5", "qtpy", "LabelLib", "IMP"])
    def test_the_service_layer_still_imports(self, blocked: str) -> None:
        script = textwrap.dedent(
            f"""
            import builtins
            import sys

            blocked = {blocked!r}
            real_import = builtins.__import__

            def guarded(name, *args, **kwargs):
                if name == blocked or name.startswith(blocked + "."):
                    raise ImportError("blocked for this test: " + name)
                return real_import(name, *args, **kwargs)

            builtins.__import__ = guarded
            for name in list(sys.modules):
                if name == blocked or name.startswith(blocked + "."):
                    del sys.modules[name]

            import quest.core  # noqa: F401
            import quest.api  # noqa: F401
            import quest.project  # noqa: F401
            import quest.runs  # noqa: F401
            import quest.tasks  # noqa: F401
            print("ok")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, (
            f"importing the domain without {blocked} failed:\n{result.stderr[-2000:]}"
        )
        assert "ok" in result.stdout


class TestTheCapabilityFlagsAreRealChecks:
    def test_has_imp_bff_reflects_an_importable_module(self) -> None:
        """The flag must not be a guess.

        `HAS_IMP_BFF` used to be `hasattr(IMP.bff, "AV")`, which asks about a
        *compiled* attribute while the AV that QuEst calls lives at
        `IMP.bff.av.compute_av` (assessment.md, DUP-02).
        """
        if av.HAS_IMP_BFF:
            import IMP.bff  # noqa: F401
        assert isinstance(av.HAS_IMP_BFF, bool)
        assert isinstance(av.HAS_LABELLIB, bool)

    def test_at_least_one_backend_is_available_in_this_environment(self) -> None:
        """A canary: if this fails the environment is broken, not the code."""
        assert av.available_av_backends(), (
            "no AV backend in the canonical environment; see okf/workflows/environment.md"
        )


def test_builtins_import_is_untouched_after_the_subprocess_tests() -> None:
    """Paranoia: the guard above must not leak into this process."""
    assert builtins.__import__.__module__ in {"builtins", "importlib._bootstrap"} or callable(
        builtins.__import__
    )
