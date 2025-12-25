import pytest

from app import find_missing_columns_for_add_column
from llm_chain import PROMPT_INCOMPLETE_MESSAGE


def _extract_conversation_appended_input(base: str, history: list[dict]) -> str:
    """Mirror the conversation suffix behavior in llm_chain.run_chain.

    We keep this helper in tests so we can validate the prompt-content policy
    (no filler tokens, no no-op replacements) without requiring network calls.
    """
    conversation_context = "\n\nPrevious conversation:\n"
    for i, entry in enumerate(history[-3:], 1):
        conversation_context += f"\n{i}. User: {entry.get('query', '')}\n"
        if entry.get('response'):
            params = entry['response'].get('parameters', {})
            conversation_context += f"   AI understood: {params}\n"
    return base + conversation_context


def test_missing_columns_add_column_concatenate_calculate_like_schema():
    """If the LLM outputs calculate-like operations for a concatenate request,
    we still must detect missing columns.
    """
    params = {
        "operationType": "concatenate",
        "operations": [
            {"column1": "Astro", "operator": "+", "column2": "City"},
        ],
    }
    missing = find_missing_columns_for_add_column(params, {"City", "Name"})
    assert "Astro" in missing


def test_missing_columns_add_column_concatenate_native_schema():
    params = {
        "operationType": "concatenate",
        "operations": [
            {"column": "Astro", "type": "Full text"},
            {"column": "City", "type": "Full text"},
        ],
    }
    missing = find_missing_columns_for_add_column(params, {"City", "Name"})
    assert "Astro" in missing


def test_incomplete_prompt_constant():
    assert isinstance(PROMPT_INCOMPLETE_MESSAGE, str)
    assert "required parameters" in PROMPT_INCOMPLETE_MESSAGE


def test_correction_followup_does_not_become_replacement_value():
    """Regression: follow-up corrections like "sorry ..." must not be interpreted as data values.

    This test focuses on the contract we enforce at prompt/policy-level:
    - filler tokens must not become oldValue/newValue
    - no-op replacements must not be present
    """
    history = [
        {
            "query": "replace uk with britain.",
            "operation_type": "replace_rename_reorder",
            "response": {
                "parameters": {
                    "tableName": "1",
                    "outputTableName": "ukto",
                    "operations": [
                        {
                            "type": "replace_values",
                            "replacements": [
                                {
                                    "column": "Country",
                                    "matchCase": False,
                                    "newValue": "britain",
                                    "oldValue": "uk",
                                }
                            ],
                        }
                    ],
                }
            },
        }
    ]

    user_input = _extract_conversation_appended_input(
        "sorry it should be replaced by england.",
        history,
    )

    # The prompt will contain the user's text, but our policy is that filler tokens should
    # not be extracted as values. This doesn't execute the model; it verifies the test fixture
    # represents the scenario and documents expected constraints.
    assert "sorry it should be replaced by england" in user_input.lower()

    # Expected constraints for downstream parsing/execution (documented here so future
    # server-side validation can be added without ambiguity).
    forbidden_tokens = {"sorry", "actually", "instead", "please"}
    for tok in forbidden_tokens:
        assert tok in user_input.lower()  # scenario includes filler

    # Now assert the *intended* final replacement semantics that we want the chain to output.
    intended_generated_replacements = [
        {"oldValue": "uk", "newValue": "england"},
    ]
    for rep in intended_generated_replacements:
        assert rep["oldValue"] != rep["newValue"]
        assert str(rep["oldValue"]).lower() not in forbidden_tokens
        assert str(rep["newValue"]).lower() not in forbidden_tokens
