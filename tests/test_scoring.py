from engagement.scoring import (DISTRACTED, DROWSY, FOCUSED, UNKNOWN, EngagementModel,
                                Features, summarize)


def run(model, feats, n=20, t0=0.0):
    r = None
    for i in range(n):
        r = model.update("STD-0001", feats, t0 + i)
    return r


def test_frontal_face_is_focused():
    r = run(EngagementModel(), Features(face=True))
    assert r.label == FOCUSED and r.score > 90 and r.confidence > 0.9


def test_phone_means_distracted():
    r = run(EngagementModel(), Features(face=True, phone=True))
    assert r.label == DISTRACTED and r.score < 60


def test_closed_eyes_means_drowsy():
    r = run(EngagementModel(), Features(face=True, ear=0.10))
    assert r.label == DROWSY


def test_looking_away_means_distracted():
    r = run(EngagementModel(), Features(face=True, yaw=60))
    assert r.label == DISTRACTED


def test_missing_face_is_unknown_then_distracted():
    m = EngagementModel()
    assert m.update("s", Features(), 0.0).label == UNKNOWN
    r = None
    for i in range(20):
        r = m.update("s", Features(), float(i))
    assert r.label == DISTRACTED


def test_declining_engagement_predicts_lower_score():
    m = EngagementModel()
    run(m, Features(face=True), n=30)
    r = run(m, Features(face=True, phone=True), n=40, t0=30)
    assert r.trend < 0 and r.predicted <= r.score


def test_summary_and_prune():
    m = EngagementModel()
    r = run(m, Features(face=True))
    s = summarize([r])
    assert s["count"] == 1 and s["counts"] == {FOCUSED: 1}
    m.prune(set())
    assert m.states == {}
