from src.monitoring import GestureConfirmation, SustainedState


def simulate_gesture(confirmation, gesture_at, duration, confirmation_time=2.5, step=0.1):
    confirmations = []
    for i in range(int(duration / step)):
        t = round(i * step, 1)
        _, confirmed = confirmation.update(gesture_at(t), t, confirmation_time)
        if confirmed:
            confirmations.append((t, confirmation.gesture))
    return confirmations


def simulate_state(state, condition_at, duration, threshold=7.0, step=0.1):
    announcements = []
    for i in range(int(duration / step)):
        t = round(i * step, 1)
        _, announce = state.update(condition_at(t), t, threshold)
        if announce:
            announcements.append(t)
    return announcements


def test_held_gesture_is_confirmed_once():
    confirmations = simulate_gesture(GestureConfirmation(grace=1.0), lambda t: "ok", 10)
    assert confirmations == [(2.5, "ok")]


def test_short_hand_loss_does_not_reset_the_gesture():
    confirmations = simulate_gesture(GestureConfirmation(grace=1.0), lambda t: None if 1.2 <= t < 1.7 else "ok", 6)
    assert confirmations == [(2.5, "ok")]


def test_long_hand_loss_resets_the_gesture():
    confirmations = simulate_gesture(GestureConfirmation(grace=1.0), lambda t: None if 1.2 <= t < 2.5 else "ok", 6)
    assert confirmations == [(5.0, "ok")]


def test_changing_the_gesture_restarts_confirmation():
    confirmations = simulate_gesture(GestureConfirmation(grace=1.0), lambda t: "two" if 3 <= t < 6 else "ok", 10)
    assert confirmations == [(2.5, "ok"), (5.5, "two"), (8.5, "ok")]


def test_alert_activates_after_threshold_and_repeats():
    state = SustainedState(repeat_interval=5, grace=1.0)
    announcements = simulate_state(state, lambda t: t < 8 and not 3 <= t < 3.5, 20)
    assert announcements[0] == 7.0
    assert len(announcements) == 3
    assert state.active


def test_active_alert_stays_until_reset():
    state = SustainedState(repeat_interval=100)
    simulate_state(state, lambda t: t < 8, 12)
    assert state.active
    state.reset()
    assert not state.active
    assert state.update(False, 13, 7.0) == (0.0, False)


def test_interrupted_state_without_grace_restarts_the_timer():
    announcements = simulate_state(SustainedState(repeat_interval=15), lambda t: t != 4.0, 13)
    assert announcements == [11.1]


def test_progress_grows_linearly():
    state = SustainedState(repeat_interval=5)
    state.update(True, 0.0, 10.0)
    progress, announce = state.update(True, 5.0, 10.0)
    assert progress == 0.5
    assert not announce
