"""ECardEnv 规则测试：全部通过 = 环境正确。"""
import numpy as np
import pytest

from env import CIVILIANS, MAX_ROUNDS, PLAY_ACE, PLAY_CIVILIAN, ECardEnv, EnvConfig


def make_env() -> ECardEnv:
    return ECardEnv(EnvConfig())


def test_reset_state_shape_and_bounds():
    env = make_env()
    s = env.reset()
    assert s.shape == (5,)
    assert s.dtype == np.float32
    assert (s >= 0).all() and (s <= 1).all()


def test_ace_ace_slave_wins():
    env = make_env()
    _, r, done, info = env.step(PLAY_ACE, PLAY_ACE)
    assert done and info["winner"] == "slave"
    assert r == -5.0


def test_single_ace_emperor_wins():
    for a_e, a_s in [(PLAY_ACE, PLAY_CIVILIAN), (PLAY_CIVILIAN, PLAY_ACE)]:
        env = make_env()
        _, r, done, info = env.step(a_e, a_s)
        assert done and info["winner"] == "emperor"
        assert r == 1.0


def test_civilian_civilian_continues():
    env = make_env()
    s, r, done, info = env.step(PLAY_CIVILIAN, PLAY_CIVILIAN)
    assert not done and r == 0.0 and info["winner"] is None
    assert s[3] == (CIVILIANS - 1) / CIVILIANS


def test_round5_forces_ace_ace():
    env = make_env()
    for _ in range(MAX_ROUNDS - 1):
        _, r, done, _ = env.step(PLAY_CIVILIAN, PLAY_CIVILIAN)
        assert not done and r == 0.0
    assert env.legal_actions("emperor") == [PLAY_ACE]
    assert env.legal_actions("slave") == [PLAY_ACE]
    _, r, done, info = env.step(PLAY_ACE, PLAY_ACE)
    assert done and info["winner"] == "slave" and r == -5.0
    assert env.round == MAX_ROUNDS


def test_illegal_action_raises():
    env = make_env()
    for _ in range(MAX_ROUNDS - 1):
        env.step(PLAY_CIVILIAN, PLAY_CIVILIAN)
    with pytest.raises(ValueError):
        env.step(PLAY_CIVILIAN, PLAY_CIVILIAN)  # 第 5 轮只能出王牌


def test_step_after_done_raises():
    env = make_env()
    env.step(PLAY_ACE, PLAY_ACE)
    with pytest.raises(RuntimeError):
        env.step(PLAY_CIVILIAN, PLAY_CIVILIAN)


def test_civilians_decrement():
    env = make_env()
    env.step(PLAY_CIVILIAN, PLAY_ACE)  # 皇帝出民，奴隶出王
    assert env.e_civ_left == CIVILIANS - 1
    assert env.s_civ_left == CIVILIANS
