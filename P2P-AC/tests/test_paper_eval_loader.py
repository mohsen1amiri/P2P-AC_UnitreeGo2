from p2pac.analysis.load import Run


def test_paper_uses_independent_evaluations(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

    # Legacy training-time results must not be used for learners.
    (eval_dir / "summary.csv").write_text(
        "t,success_rate\n50000,1.0\n"
    )
    (eval_dir / "final_episodes.csv").write_text(
        "success\n1\n"
    )

    learner = Run(
        root=str(tmp_path),
        config={"method": "p2p", "algo": "sac"},
    )

    assert learner.eval_summary().empty
    assert learner.final_episodes().empty

    # Independent snapshot evaluations.
    curve_dir = eval_dir / "reeval_curve10"
    curve_dir.mkdir()
    (curve_dir / "summary.csv").write_text(
        "t,success_rate\n50000,0.4\n"
    )

    # Independent final evaluation.
    final_dir = eval_dir / "reeval_final100"
    final_dir.mkdir()
    (final_dir / "episodes.csv").write_text(
        "success\n0\n"
    )

    assert learner.eval_summary()["success_rate"].tolist() == [0.4]
    assert learner.final_episodes()["success"].tolist() == [0]

    # Behavior cloning must also use independent final evaluation.
    bc = Run(root=str(tmp_path), config={"method": "bc"})
    assert bc.final_episodes()["success"].tolist() == [0]

    # Planner-only references retain their direct final evaluation.
    planner = Run(root=str(tmp_path), config={"method": "planner"})
    assert planner.final_episodes()["success"].tolist() == [1]