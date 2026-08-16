import math

from train.stage4.evaluate_all import (
    _select_trajectory_rows,
    calc_dtw,
    calc_l2_distance,
    calc_waypoint_loss,
    parse_trajectory,
    summarize_predictions,
)


def test_parse_trajectory_extracts_coordinate_list():
    parsed = parse_trajectory("reasoning... [[1,2],[3,4],[5,6],[7,8],[9,10]] done")
    assert parsed == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]


def test_waypoint_loss_matches_stage4_normalized_squared_euclidean():
    ground_truth = [[0.0, 0.0]] * 5
    prediction = [[1000.0, 0.0]] * 5

    assert calc_waypoint_loss(ground_truth, prediction) == 1.0
    assert calc_l2_distance(ground_truth, prediction) == 1000.0
    assert calc_dtw(ground_truth, prediction) == 5000.0


def test_invalid_waypoint_count_is_reported_as_failure():
    ground_truth = [[[0.0, 0.0]] * 5]
    prediction = [[[0.0, 0.0]] * 4]

    assert math.isnan(calc_waypoint_loss(ground_truth[0], prediction[0]))
    summary = summarize_predictions(ground_truth, prediction)
    assert summary["aggregate"]["valid_samples"] == 0
    assert summary["aggregate"]["failed_samples"] == 1


class _FakeDataset:
    column_names = ["assistant", "other"]

    def __init__(self, rows, selected=None):
        self.rows = rows
        self.selected = selected

    def select_columns(self, _columns):
        return self

    def select(self, indices):
        return _FakeDataset(self.rows, list(indices))

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)


def test_select_trajectory_rows_skips_non_trajectory_answers():
    valid = "[[1,2],[3,4],[5,6],[7,8],[9,10]]"
    dataset = _FakeDataset(
        [
            {"assistant": "plain QA answer"},
            {"assistant": valid},
            {"assistant": "[[1,2],[3,4]]"},
            {"assistant": valid},
        ]
    )

    selected, indices = _select_trajectory_rows(dataset, 2)

    assert indices == [1, 3]
    assert selected.selected == [1, 3]
