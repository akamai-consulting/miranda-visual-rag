import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from grounding import interpret, GroundingError, apply_tumble_dry_mask


def result(**changes):
    base = dict(domain='fashion_search', query_en='grey zip sweater',
                is_followup=False, decision='SEARCH', tumble_dry='NONE', reason='')
    base.update(changes)
    return base


class GroundingTests(unittest.TestCase):
    def test_normal_search(self):
        x = interpret('grey zip sweater', None, lambda p: result())
        self.assertEqual(x['tumble_dry'], 'NONE')

    def test_hard_yes(self):
        x = interpret('must be tumble-dryable', None,
                      lambda p: result(tumble_dry='REQUIRED'))
        self.assertEqual(x['tumble_dry'], 'REQUIRED')

    def test_hard_no(self):
        x = interpret('must not be tumble-dried', None,
                      lambda p: result(tumble_dry='FORBIDDEN'))
        self.assertEqual(x['tumble_dry'], 'FORBIDDEN')

    def test_unsupported(self):
        x = interpret('not black', None,
                      lambda p: result(decision='UNSUPPORTED', reason='no color exclusion metadata'))
        self.assertEqual(x['decision'], 'UNSUPPORTED')

    def test_invalid_state_fails_closed(self):
        with self.assertRaises(GroundingError):
            interpret('must be tumble dryable', None,
                      lambda p: result(tumble_dry='MAYBE'))

    def test_missing_fields_fails_closed(self):
        with self.assertRaises(GroundingError):
            interpret('test', None, lambda p: {'domain':'fashion_search'})

    def test_care_terms_must_not_leak_into_clip(self):
        with self.assertRaises(GroundingError):
            interpret('dryer safe sweater', None,
                      lambda p: result(tumble_dry='REQUIRED', query_en='dryer safe sweater'))

    def test_mask_before_topk(self):
        scores = torch.tensor([[.326, .321, .200]])
        paths = ['archive/Grey_Cardigan/img.jpg', 'archive/Grey_Sweater/img.jpg',
                 'archive/Wool_Coat/img.jpg']
        masked, diag = apply_tumble_dry_mask(scores, paths, 'REQUIRED')
        self.assertEqual(torch.topk(masked, 1).indices.item(), 1)
        self.assertEqual(diag['rejected'], 2)
        self.assertEqual(diag['eligible'], 1)
        forbidden, _ = apply_tumble_dry_mask(scores, paths, 'FORBIDDEN')
        self.assertEqual(torch.topk(forbidden, 1).indices.item(), 0)

    def test_no_constraint_preserves_scores(self):
        scores = torch.tensor([[.3, .2]])
        result_scores, diag = apply_tumble_dry_mask(scores, ['a', 'b'], 'NONE')
        self.assertIs(result_scores, scores)
        self.assertIsNone(diag)

    def test_mismatch_fails_closed(self):
        with self.assertRaises(GroundingError):
            apply_tumble_dry_mask(torch.ones(1, 3), ['one'], 'REQUIRED')


if __name__ == '__main__':
    unittest.main()
