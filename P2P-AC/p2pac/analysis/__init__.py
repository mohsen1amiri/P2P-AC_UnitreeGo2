"""Aggregation of raw run logs into the statistics, tables and figure data of the paper.

Everything in this package reads only the run directories (no simulator, no retraining):

* :mod:`p2pac.analysis.load`        -- discovery and loading of runs;
* :mod:`p2pac.analysis.stats`       -- N80 with censoring, bootstrap confidence intervals, Wilson intervals, tests;
* :mod:`p2pac.analysis.aggregate`   -- ``python -m p2pac.analysis.aggregate --spec configs/analysis/paper.yaml``;
* :mod:`p2pac.analysis.theory`      -- empirical checks of Assumptions 1-2, Proposition 2, Corollary 2, Proposition 3;
* :mod:`p2pac.analysis.planner_gap` -- suboptimality gap of the planner versus its budget (Corollary 1).
"""
