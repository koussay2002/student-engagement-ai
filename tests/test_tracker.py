from engagement.tracker import IoUTracker, iou


def test_iou_identical_and_disjoint():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_ids_are_stable_across_frames():
    t = IoUTracker(min_hits=2)
    assert t.update([(0, 0, 100, 200), (300, 0, 400, 200)]) == []
    out = t.update([(2, 1, 102, 201), (301, 2, 401, 202)])
    assert [x.sid for x in out] == ["STD-0001", "STD-0002"]
    out = t.update([(4, 2, 104, 202), (302, 3, 402, 203)])
    assert [x.sid for x in out] == ["STD-0001", "STD-0002"]


def test_lost_tracks_are_dropped():
    t = IoUTracker(max_missed=2)
    t.update([(0, 0, 50, 50)])
    for _ in range(4):
        t.update([])
    assert t.tracks == []
