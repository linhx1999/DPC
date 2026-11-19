"""Offline regressions for output limits, cost records, and agent fallback."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from baseline.token_usage import summarize, usage_recorder
from dpc.agents.base_agent import BaseAgent
from dpc.agents.selector_agent import EquivalenceGrouperAgent
from dpc.agents.slicer_agent import SlicerAgent
from dpc.agents.solver_agent import PythonSolverAgent
from dpc.agents.tester_agent import TesterAgent as DataTesterAgent
from dpc.core.pipeline import DPCPipeline
from dpc.llm.errors import EmptyResponseError, OutputTokenLimitError
from dpc.llm.openai_llm import OpenAILLM


def response(content="ok", finish_reason="stop", tokens=20):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=tokens, total_tokens=tokens + 10),
        choices=[SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(content=content, reasoning="reasoning is not final text"),
        )],
    )


class OutputBudgetTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name)
        env = patch.dict(os.environ, {
            "DPC_USAGE_DIR": str(self.output), "DPC_SESSION": str(self.output / "session"),
        })
        env.start()
        self.addCleanup(env.stop)
        client = patch("dpc.llm.openai_llm.openai.OpenAI")
        self.client = client.start().return_value
        self.addCleanup(client.stop)
        self.create = self.client.chat.completions.create
        self.llm = OpenAILLM(
            model_name="test-model", api_key="test-key", max_tokens=16384,
            usage_recorder=usage_recorder("selection", "1", "test-model"),
        )

    def records(self):
        return [json.loads(p.read_text()) for p in (self.output / "telemetry/calls").glob("*.json")]

    def test_complete_response_is_returned_and_counted(self):
        self.create.return_value = response("  final answer  ")
        self.assertEqual(self.llm.ask([]), "final answer")
        record, = self.records()
        self.assertEqual(record["outcome"], "success")
        self.assertEqual(record["finish_reason"], "stop")
        self.assertEqual(record["max_tokens"], 16384)
        self.assertEqual(record["total_tokens"], 30)
        self.assertIsNone(record["error_type"])

    def test_length_failure_is_recorded_once_without_retry(self):
        self.create.return_value = response(None, "length", 16384)
        with patch("dpc.llm.openai_llm.time.sleep") as sleep:
            with self.assertRaises(OutputTokenLimitError):
                self.llm.ask([])
        self.assertEqual(self.create.call_count, 1)
        sleep.assert_not_called()
        record, = self.records()
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["outcome"], "output_limit")
        self.assertEqual(record["finish_reason"], "length")
        self.assertEqual(record["response_chars"], 0)
        self.assertEqual(record["error_type"], "OutputTokenLimitError")
        self.assertEqual(self.llm.get_usage()["completion_tokens"], 16384)
        summarize(self.output)
        total = json.loads((self.output / "metrics/summary.json").read_text())["total"]
        self.assertEqual(total["total_tokens"], 16394)
        self.assertEqual(total["outcome_counts"], {"output_limit": 1})

    def test_partial_text_with_length_is_not_accepted(self):
        self.create.return_value = response('{"test_data": {}}', "length", 16384)
        with self.assertRaises(OutputTokenLimitError):
            self.llm.ask([])
        self.assertEqual(self.create.call_count, 1)

    def test_empty_final_text_does_not_use_reasoning_or_retry(self):
        for content in [None, "", " \n"]:
            with self.subTest(content=content):
                self.create.reset_mock()
                self.create.return_value = response(content)
                with self.assertRaises(EmptyResponseError):
                    self.llm.ask([])
                self.assertEqual(self.create.call_count, 1)
        self.assertTrue(all(r["outcome"] == "empty_response" for r in self.records()))

    def test_transient_failure_keeps_retry_and_cost_accounting(self):
        self.create.side_effect = [ConnectionError("temporary connection failure"), response("ok")]
        with patch("dpc.llm.openai_llm.time.sleep") as sleep:
            self.assertEqual(self.llm.ask([]), "ok")
        self.assertEqual(self.create.call_count, 2)
        sleep.assert_called_once_with(2)
        records = sorted(self.records(), key=lambda r: r["attempt"])
        self.assertEqual([r["outcome"] for r in records], ["api_error", "success"])
        self.assertEqual(records[0]["error_type"], "ConnectionError")
        self.assertIsNone(records[0]["total_tokens"])
        summarize(self.output)
        total = json.loads((self.output / "metrics/summary.json").read_text())["total"]
        self.assertEqual(total["calls_missing_usage"], 1)
        self.assertEqual(total["total_tokens"], 30)

    def test_legacy_records_are_not_reclassified_as_truncation(self):
        usage_recorder("selection", "1", "test-model")(response(tokens=4096).usage, "failed")
        summarize(self.output)
        total = json.loads((self.output / "metrics/summary.json").read_text())["total"]
        self.assertEqual(total["outcome_counts"], {"unclassified": 1})
        self.assertEqual(total["total_tokens"], 4106)


class JsonAgent(BaseAgent):
    def run(self):
        return self._ask_and_parse_json([], max_retries=3)


class AgentResponseFailureTest(unittest.TestCase):
    def test_response_failures_bypass_all_agent_correction_loops(self):
        cases = [
            (JsonAgent, {}),
            (SlicerAgent, {"candidate_sqls": ["SELECT 1"], "full_schema": {}}),
            (DataTesterAgent, {"question": "q", "sql_1": "SELECT 1", "sql_2": "SELECT 2", "sliced_schema": {}}),
            (PythonSolverAgent, {"question": "q", "test_data": {}, "sliced_schema": {}}),
            (EquivalenceGrouperAgent, {
                "question": "q", "candidate_sqls": ["SELECT 1", "SELECT 2"],
                "full_schema": {}, "num_grouping_attempts": 3,
            }),
        ]
        for agent_type, kwargs in cases:
            for error in [OutputTokenLimitError(16384), EmptyResponseError()]:
                with self.subTest(agent=agent_type.__name__, error=type(error).__name__):
                    llm = Mock()
                    llm.ask.side_effect = error
                    agent = agent_type(llm)
                    with self.assertRaises(type(error)):
                        agent.run(**kwargs)
                    self.assertEqual(llm.ask.call_count, 1)
                    self.assertEqual(agent.corrections, 0)

    def test_json_format_errors_still_allow_correction(self):
        llm = Mock()
        llm.ask.side_effect = ["not json", '{"valid": true}']
        self.assertEqual(JsonAgent(llm).run(), {"valid": True})
        self.assertEqual(llm.ask.call_count, 2)

    def test_pipeline_falls_back_to_champion_on_slicer_output_limit(self):
        llm = Mock()
        llm.ask.side_effect = OutputTokenLimitError(16384)
        llm.get_usage.return_value = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tester, solver = Mock(), Mock()
        pipeline = DPCPipeline(SlicerAgent(llm), tester, solver, llm=llm)
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.sqlite"
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE items (n INTEGER)")
                conn.execute("INSERT INTO items VALUES (1)")
            result = pipeline.run("q", str(db), ["SELECT n FROM items", "SELECT n+1 FROM items"])
        self.assertEqual(result["selected_sql"], "SELECT n FROM items")
        self.assertIn("finish_reason=length", result["selection_reason"])
        self.assertEqual(llm.ask.call_count, 1)
        tester.run.assert_not_called()
        solver.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
