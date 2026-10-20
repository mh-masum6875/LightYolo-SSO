import os, sys, tempfile, sqlite3
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.search import space
from src.search.optimizers import make, STRATEGIES
from src.search.fitness import MockBackend, FitnessLogger, compose_J
from src.search.run import run_search
from src.search.analyze import best_so_far, evals_to_target

NON_TPE = [s for s in STRATEGIES if s != "tpe"]

def _has_optuna():
    try:
        import optuna  # noqa
        return True
    except ImportError:
        return False

def test_all_strategies_respect_budget_and_bounds():
    """Master Plan 6.2 acceptance: equal evaluation budgets, provable."""
    budget = 47   # deliberately not a multiple of pop size
    for s in NON_TPE + (["tpe"] if _has_optuna() else []):
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "p.sqlite")
            fit = FitnessLogger(MockBackend(), db, s, 0)
            theta, J = run_search(s, 0, budget, fit)
            assert fit.n_evals == budget, f"{s}: {fit.n_evals} != {budget}"
            assert theta is not None and np.isfinite(J)
            t = space.discretize(theta)
            assert np.all(t >= space.LOW) and np.all(t <= space.HIGH)
            con = sqlite3.connect(db)
            n = con.execute("SELECT COUNT(*) FROM runs WHERE metric='J'").fetchone()[0]
            con.close()
            assert n == budget, f"{s}: db has {n} J rows, expected {budget}"

def test_same_seed_same_trajectory():
    for s in NON_TPE:
        with tempfile.TemporaryDirectory() as td:
            f1 = FitnessLogger(MockBackend(), os.path.join(td, "a.sqlite"), s, 3)
            f2 = FitnessLogger(MockBackend(), os.path.join(td, "b.sqlite"), s, 3)
            run_search(s, 3, 40, f1)
            run_search(s, 3, 40, f2)
            assert f1.history == f2.history, f"{s} not deterministic"

def test_different_seeds_differ():
    with tempfile.TemporaryDirectory() as td:
        f1 = FitnessLogger(MockBackend(), os.path.join(td, "a.sqlite"), "sso", 0)
        f2 = FitnessLogger(MockBackend(), os.path.join(td, "b.sqlite"), "sso", 1)
        run_search("sso", 0, 40, f1)
        run_search("sso", 1, 40, f2)
        assert f1.history != f2.history

def test_optimizers_beat_first_guess():
    """Sanity: after 100 evals every strategy improved on its first evaluation."""
    for s in NON_TPE:
        with tempfile.TemporaryDirectory() as td:
            fit = FitnessLogger(MockBackend(), os.path.join(td, "p.sqlite"), s, 0)
            _, J = run_search(s, 0, 100, fit)
            assert J < fit.history[0][1], f"{s} never improved"

def test_compose_J_penalty():
    small = compose_J(0.9, 0.8, 1.5, 20.0)
    big = compose_J(0.9, 3.0, 4.0, 20.0)   # violates both constraints
    assert big > small + 10.0

def test_discretize_is_canonical():
    t = np.array([31.7, .5, .5, .5, 1., 63.2, .5, .5, 1., 1., 1., 2.6, 1.4, .7, .2])
    d = space.discretize(t)
    assert d[0] == 32 and d[5] == 63 and d[11] == 3 and d[12] == 1
    assert d[13] == 1.0 and d[14] == 0.0
    assert np.array_equal(d, space.discretize(d))   # idempotent

def test_evals_to_target():
    js = [5.0, 4.0, 3.0, 1.005, 2.0, 1.0]
    assert evals_to_target(js, tol=0.01) == 4       # 1.005 <= 1.0 + 0.01
    assert best_so_far(js) == [5.0, 4.0, 3.0, 1.005, 1.005, 1.0]

def test_revol_population_is_fifo():
    from src.search.optimizers import RegularizedEvolution
    o = RegularizedEvolution(seed=0, pop_size=5, tournament=3)
    init = o.ask()
    o.tell(init, list(range(len(init))))
    assert len(o.population) == 5
    first = o.population[0][0].copy()
    child = o.ask()
    o.tell(child, [99.0])                            # bad child still enters, oldest leaves
    assert len(o.population) == 5
    assert not any(np.array_equal(first, p) for p, _ in o.population)
