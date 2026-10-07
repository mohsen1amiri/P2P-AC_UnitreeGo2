import numpy as np
import pytest

from p2pac.task import Outcome, NavigationTask, TaskConfig


@pytest.fixture
def task():
    return NavigationTask(TaskConfig())


def test_goal_and_failure_sets_are_disjoint(task):
    g = task.cfg.goal
    assert task.distance_to_goal(g) == 0
    assert not task.in_obstacle(g) and not task.out_of_bounds(g)


def test_spawn_distribution(task):
    rng = np.random.default_rng(0)
    for _ in range(500):
        xy = task.sample_start(rng)
        assert task.is_valid_start(xy)
        assert xy[1] <= task.cfg.spawn_y_max
        assert task.obstacle_clearance(xy) > task.cfg.spawn_clearance - 1e-9
        assert task.wall_clearance(xy) >= 0


def test_fixed_eval_starts_are_reproducible(task):
    a = task.sample_starts(20, seed=7)
    b = task.sample_starts(20, seed=7)
    np.testing.assert_array_equal(a, b)


def test_classification_priorities(task):
    g = task.cfg.goal
    assert task.classify(g, height=0.3, contact=True, step_count=1) == Outcome.BASE_CONTACT
    assert task.classify(g, height=0.05, contact=False, step_count=1) == Outcome.FALL
    o = task.cfg.obstacle_array[0, :2]
    assert task.classify(o, height=0.3, contact=False, step_count=1) == Outcome.COLLISION_OBSTACLE
    assert task.classify(np.array([1.9, -1.0]), height=0.3, contact=False, step_count=1) == Outcome.OUT_OF_BOUNDS
    assert task.classify(g, height=0.3, contact=False, step_count=1) == Outcome.GOAL
    free = np.array([-1.0, -1.5])
    assert task.classify(free, height=0.3, contact=False, step_count=task.cfg.max_episode_steps) == Outcome.TIMEOUT
    assert Outcome.TIMEOUT.is_terminal is False and Outcome.GOAL.is_terminal and Outcome.FALL.is_terminal


def test_reward_eq1():
    task = NavigationTask(TaskConfig())
    u = np.array([0.3, -0.4])  # ||u||^2 = 0.25
    r0 = -1.0 - 0.02 * 0.25
    assert task.reward(Outcome.RUNNING, u).reward == pytest.approx(r0)
    assert task.reward(Outcome.GOAL, u).reward == pytest.approx(r0 + 100.0)
    assert task.reward(Outcome.COLLISION_OBSTACLE, u).reward == pytest.approx(r0 - 200.0)
    assert task.reward(Outcome.TIMEOUT, u).reward == pytest.approx(r0)
    assert task.reward(Outcome.GOAL, u).terminated and task.reward(Outcome.TIMEOUT, u).truncated
