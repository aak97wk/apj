
import jittor as jt
from jittor import init
from jittor import nn
from jittor_implementations import one_hot


'Argmax matcher implementation.\n\nThis class takes a similarity matrix and matches columns to rows based on the\nmaximum value per column. One can specify matched_thresholds and\nto prevent columns from matching to rows (generally resulting in a negative\ntraining example) and unmatched_theshold to ignore the match (generally\nresulting in neither a positive or negative training example).\n\nThis matcher is used in Fast(er)-RCNN.\n\nNote: matchers are used in TargetAssigners. There is a create_target_assigner\nfactory function for popular implementations.\n'
from . import matcher

class ArgMaxMatcher(matcher.Matcher):
    'Matcher based on highest value.\n\n    This class computes matches from a similarity matrix. Each column is matched\n    to a single row.\n\n    To support object detection target assignment this class enables setting both\n    matched_threshold (upper threshold) and unmatched_threshold (lower thresholds)\n    defining three categories of similarity which define whether examples are\n    positive, negative, or ignored:\n    (1) similarity >= matched_threshold: Highest similarity. Matched/Positive!\n    (2) matched_threshold > similarity >= unmatched_threshold: Medium similarity.\n            Depending on negatives_lower_than_unmatched, this is either\n            Unmatched/Negative OR Ignore.\n    (3) unmatched_threshold > similarity: Lowest similarity. Depending on flag\n            negatives_lower_than_unmatched, either Unmatched/Negative OR Ignore.\n    For ignored matches this class sets the values in the Match object to -2.\n    '

    def __init__(self, matched_threshold, unmatched_threshold=None, negatives_lower_than_unmatched=True, force_match_for_each_row=False):
        'Construct ArgMaxMatcher.\n\n        Args:\n            matched_threshold: Threshold for positive matches. Positive if\n                sim >= matched_threshold, where sim is the maximum value of the\n                similarity matrix for a given column. Set to None for no threshold.\n            unmatched_threshold: Threshold for negative matches. Negative if\n                sim < unmatched_threshold. Defaults to matched_threshold\n                when set to None.\n            negatives_lower_than_unmatched: Boolean which defaults to True. If True\n                then negative matches are the ones below the unmatched_threshold,\n                whereas ignored matches are in between the matched and unmatched\n                threshold. If False, then negative matches are in between the matched\n                and unmatched threshold, and everything lower than unmatched is ignored.\n            force_match_for_each_row: If True, ensures that each row is matched to\n                at least one column (which is not guaranteed otherwise if the\n                matched_threshold is high). Defaults to False. See\n                argmax_matcher_test.testMatcherForceMatch() for an example.\n\n        Raises:\n            ValueError: if unmatched_threshold is set but matched_threshold is not set\n                or if unmatched_threshold > matched_threshold.\n        '
        if ((matched_threshold is None) and (unmatched_threshold is not None)):
            raise ValueError('Need to also define matched_threshold when unmatched_threshold is defined')
        self._matched_threshold = matched_threshold
        if (unmatched_threshold is None):
            self._unmatched_threshold = matched_threshold
        else:
            if (unmatched_threshold > matched_threshold):
                raise ValueError('unmatched_threshold needs to be smaller or equal to matched_threshold')
            self._unmatched_threshold = unmatched_threshold
        if (not negatives_lower_than_unmatched):
            if (self._unmatched_threshold == self._matched_threshold):
                raise ValueError('When negatives are in between matched and unmatched thresholds, these cannot be of equal value. matched: %s, unmatched: %s', self._matched_threshold, self._unmatched_threshold)
        self._force_match_for_each_row = force_match_for_each_row
        self._negatives_lower_than_unmatched = negatives_lower_than_unmatched

    def _match(self, similarity_matrix):
        'Tries to match each column of the similarity matrix to a row.\n\n        Args:\n            similarity_matrix: tensor of shape [N, M] representing any similarity metric.\n\n        Returns:\n            Match object with corresponding matches for each of M columns.\n        '

        def _match_when_rows_are_empty():
            "Performs matching when the rows of similarity matrix are empty.\n\n            When the rows are empty, all detections are false positives. So we return\n            a tensor of -1's to indicate that the columns do not match to any rows.\n\n            Returns:\n                matches:  int32 tensor indicating the row each column matches to.\n            "
            return ((- 1) * jt.ones(similarity_matrix.shape[1], dtype=jt.long))

        def _match_when_rows_are_non_empty():
            'Performs matching when the rows of similarity matrix are non empty.\n\n            Returns:\n                matches:  int32 tensor indicating the row each column matches to.\n            '
            matches = jt.argmax(similarity_matrix, dim=0)
            if (self._matched_threshold is not None):
                matched_vals = jt.max(similarity_matrix, dim=0)[0]
                below_unmatched_threshold = (self._unmatched_threshold > matched_vals)
                between_thresholds = ((matched_vals >= self._unmatched_threshold) & (self._matched_threshold > matched_vals))
                if self._negatives_lower_than_unmatched:
                    matches = self._set_values_using_indicator(matches, below_unmatched_threshold, (- 1))
                    matches = self._set_values_using_indicator(matches, between_thresholds, (- 2))
                else:
                    matches = self._set_values_using_indicator(matches, below_unmatched_threshold, (- 2))
                    matches = self._set_values_using_indicator(matches, between_thresholds, (- 1))
            if self._force_match_for_each_row:
                force_match_column_ids = jt.argmax(similarity_matrix, dim=1)
                force_match_column_indicators = one_hot(force_match_column_ids, similarity_matrix.shape[1])
                force_match_row_ids = jt.argmax(force_match_column_indicators, dim=0)
                force_match_column_mask = jt.max(force_match_column_indicators, dim=0)[0].bool()
                final_matches = jt.where(force_match_column_mask, x=force_match_row_ids, y=matches)
                return final_matches
            else:
                return matches
        if (similarity_matrix.shape[0] == 0):
            return _match_when_rows_are_empty()
        else:
            return _match_when_rows_are_non_empty()

    def _set_values_using_indicator(self, x, indicator, val):
        'Set the indicated fields of x to val.\n\n        Args:\n            x: tensor.\n            indicator: boolean with same shape as x.\n            val: scalar with value to set.\n\n        Returns:\n            modified tensor.\n        '
        indicator = indicator.astype(x.dtype)
        return ((x * (1 - indicator)) + (val * indicator))
