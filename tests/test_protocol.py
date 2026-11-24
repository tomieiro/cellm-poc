import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from celnn import DifferentiableCellularNetwork

from cellm_poc.model import CellLM, Shape
from cellm_poc.protocol import boundary, prepare, recall_cases, train_arm


class ContinuityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.shape = Shape(
            block=8, width=12, steps=2, radius=2, lamp_radius=2, lamp_width=4
        )
        self.model = CellLM(32, self.shape)
        self.ids = torch.randint(32, (2, 24))

    def test_celnn_advances_each_cellular_step_and_trains_templates(self):
        self.assertIsInstance(
            getattr(self.model, "dynamics", None), DifferentiableCellularNetwork
        )
        dynamics = self.model.dynamics
        with patch.object(dynamics, "step", wraps=dynamics.step) as step:
            output = self.model.evolve(self.ids[:, :8])
        self.assertEqual(step.call_count, self.shape.steps)
        output.square().mean().backward()
        self.assertGreater(float(dynamics.control.grad.abs().sum()), 0.0)
        self.assertGreater(float(dynamics.feedback.grad.abs().sum()), 0.0)

    @torch.no_grad()
    def test_reset_erases_history_and_carry_uses_it(self):
        changed = self.ids.clone()
        changed[:, :8] = (changed[:, :8] + 1) % 32
        torch.testing.assert_close(
            self.model(self.ids, "reset")[:, 8:], self.model(changed, "reset")[:, 8:]
        )
        torch.testing.assert_close(
            self.model(self.ids, "carry")[:, :8], self.model(self.ids, "reset")[:, :8]
        )
        self.assertGreater(
            float(
                (
                    self.model(self.ids, "carry")[:, 8:]
                    - self.model(changed, "carry")[:, 8:]
                )
                .abs()
                .max()
            ),
            1e-5,
        )

    def test_lamp_has_no_message_without_a_preceding_cell(self):
        activities = torch.randn(2, 8, self.shape.width)
        actual = self.model.local_drive(activities)
        torch.testing.assert_close(actual[:, 0], torch.zeros_like(actual[:, 0]))
        self.assertGreater(float(actual[:, 1:].detach().abs().max()), 0.0)

    def test_future_tokens_do_not_change_predictions_in_either_arm(self):
        changed = self.ids.clone()
        changed[:, 11:] = (changed[:, 11:] + 1) % 32
        for mode in ("carry", "reset"):
            torch.testing.assert_close(
                self.model(self.ids, mode)[:, :11], self.model(changed, mode)[:, :11]
            )

    @torch.no_grad()
    def test_overlap_scores_the_same_targets_without_state(self):
        rows = boundary(self.model, self.ids[0].tolist(), overlap=4, device="cpu")
        logit, _ = self.model.block(self.ids[:1, 4:12])
        target = self.ids[0, 9:13]
        expected = torch.nn.functional.cross_entropy(logit[0, 4:], target)
        self.assertEqual(sum(row["targets"] for row in rows), 8)
        self.assertEqual(rows[0]["position"], 8)
        self.assertAlmostEqual(rows[0]["overlap_nll"], float(expected), places=6)

    def test_counterfactual_changes_only_the_randomized_value(self):
        pairs = recall_cases(
            [1, 2], [3], [4, 5, 6, 7, 8, 9, 10, 14], [11, 12, 13], [4, 10], 4, 16, 91
        )
        self.assertEqual(len(pairs), 8)
        self.assertEqual(
            pairs,
            recall_cases(
                [1, 2],
                [3],
                [4, 5, 6, 7, 8, 9, 10, 14],
                [11, 12, 13],
                [4, 10],
                4,
                16,
                91,
            ),
        )
        for distance, prompt, alternative, value, other in pairs:
            self.assertEqual(
                [i for i, (a, b) in enumerate(zip(prompt, alternative)) if a != b], [2]
            )
            self.assertEqual((prompt[2], alternative[2]), (value, other))
            self.assertEqual(len(prompt) - 2, distance)

    def test_paired_training_uses_identical_batches_and_exact_budgets(self):
        docs = [list(range(i, i + 17)) for i in range(8)]
        runs = []
        for arm in ("carry", "reset"):
            torch.manual_seed(11)
            model = CellLM(32, self.shape)
            result = train_arm(
                model, arm, docs, docs[-2:], [31, 65], 2, 91, "cpu", 3e-4
            )
            runs.append(result)
            self.assertEqual(
                [row["trained_tokens"] for row in result["learning_curve"]], [31, 65]
            )
            self.assertTrue(
                all(math.isfinite(row["nll"]) for row in result["learning_curve"])
            )
        self.assertEqual(
            runs[0]["initial_weights_sha256"], runs[1]["initial_weights_sha256"]
        )
        self.assertEqual(runs[0]["batch_sha256"], runs[1]["batch_sha256"])
        self.assertNotEqual(
            runs[0]["final_weights_sha256"], runs[1]["final_weights_sha256"]
        )

    def test_duplicate_documents_across_splits_are_rejected(self):
        with TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{i}.jsonl" for i in range(3)]
            for path in paths:
                path.write_text('{"text":"The same document."}\n')
            with self.assertRaisesRegex(ValueError, "duplicado"):
                prepare(paths, vocab_size=300, context=16)


if __name__ == "__main__":
    unittest.main()
