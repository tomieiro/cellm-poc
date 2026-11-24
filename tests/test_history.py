import unittest

from cellm_poc.history import summarize_lamp


class HistoricalResultsTest(unittest.TestCase):
    def test_lamp_uses_source_weights_and_same_targets(self):
        def row(full, off):
            return {
                "full": {"loss_nats": full, "tokens": 2047},
                "local_off": {"loss_nats": off, "tokens": 2047},
            }

        measure = summarize_lamp(
            {"a": [row(2, 3), row(4, 5)], "b": [row(7, 9)]}, {"a": 0.75, "b": 0.25}
        )
        self.assertEqual(measure["windows"], 3)
        self.assertEqual(measure["targets_per_arm"], 6141)
        self.assertEqual(measure["full_nll"], 4.0)
        self.assertEqual(measure["local_off_nll"], 5.25)

    def test_d4_recomputes_bpc_with_historical_utf8_denominator(self):
        from math import log

        from cellm_poc.history import summarize_d4

        row = {
            "target_tokens": 50_000_010,
            "held_out": {"nll": 2 * log(2), "bpc": 1.0, "tokens": 12, "utf8_bytes": 24},
        }
        result = summarize_d4({"cellm": row, "wide": row, "deep": row})
        self.assertEqual(result["cellm"]["bpc"], 1.0)
        self.assertEqual(result["cellm"]["trained_tokens"], 50_000_010)
        self.assertEqual(result["cellm"]["eval_utf8_bytes"], 24)
