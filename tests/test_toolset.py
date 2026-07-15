from __future__ import annotations

import pytest
from pydantic_ai.toolsets import FunctionToolset

from infolang_pydantic_ai import infolang_toolset


def test_default_toolset_has_all_four_tools() -> None:
    toolset = infolang_toolset()
    assert isinstance(toolset, FunctionToolset)
    assert set(toolset.tools) == {"recall", "investigate", "remember", "forget"}


def test_tools_have_model_facing_descriptions() -> None:
    toolset = infolang_toolset()
    for name, tool in toolset.tools.items():
        assert tool.description, f"{name} is missing a description"


def test_include_selects_subset() -> None:
    toolset = infolang_toolset(include=["recall", "remember"])
    assert set(toolset.tools) == {"recall", "remember"}


def test_include_single_tool() -> None:
    toolset = infolang_toolset(include=["forget"])
    assert set(toolset.tools) == {"forget"}


def test_include_unknown_tool_raises() -> None:
    with pytest.raises(ValueError, match="Unknown InfoLang tool"):
        infolang_toolset(include=["recall", "bogus"])


def test_include_empty_raises() -> None:
    with pytest.raises(ValueError, match="no tools"):
        infolang_toolset(include=[])


def test_toolset_kwargs_forwarded() -> None:
    toolset = infolang_toolset(max_retries=7)
    assert toolset.max_retries == 7


def test_unknown_toolset_kwarg_raises_type_error() -> None:
    with pytest.raises(TypeError):
        infolang_toolset(not_a_real_kwarg=1)  # type: ignore[call-overload]
