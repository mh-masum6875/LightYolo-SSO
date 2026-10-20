"""WP2 pluggable optimizers behind one ask/tell interface (Master Plan 6.1).

class Optimizer(Protocol):
    def ask(self) -> list[np.ndarray]: ...
    def tell(self, thetas, fitnesses) -> None: ...
    def best(self) -> tuple[np.ndarray, float]: ...

All optimizers MINIMIZE. All use a private np.random.Generator seeded at
construction, so a (strategy, seed) pair is fully deterministic.
Strategies (Master Plan 6.2): sso (faithful legacy port, WP0-audited),
pso (inertia weight w=0.72, c1=c2=1.49), random, revol (regularised
evolution, tournament 5, one-dimension mutation), tpe (Optuna).
"""
from __future__ import annotations
from typing import Protocol
import numpy as np
from . import space

class Optimizer(Protocol):
    def ask(self) -> list[np.ndarray]: ...
    def tell(self, thetas: list[np.ndarray], fitnesses: list[float]) -> None: ...
    def best(self) -> tuple[np.ndarray, float]: ...

class _Base:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self._best_theta: np.ndarray | None = None
        self._best_f = float("inf")

    def _update_best(self, thetas, fitnesses):
        for t, f in zip(thetas, fitnesses):
            if f < self._best_f:
                self._best_f = float(f)
                self._best_theta = np.asarray(t, dtype=float).copy()

    def best(self):
        return self._best_theta, self._best_f

class SSO(_Base):
    """Standard Shehadeh SSO, faithful port of the legacy implementation
    (WP0 eq2 audit: log10 coefficients, rho1/rho2 ~ U(7,14), rho3 ~ U(35.5,38.5),
    shared attraction coefficient on pbest and gbest terms, r1 reused as the
    outer inertia multiplier). No algorithmic modification."""
    VEL_COEFF = 0.3
    LOG_MIN, LOG_MAX = 7.0, 14.0
    POS_LOG_MIN, POS_LOG_MAX = 35.5, 38.5

    def __init__(self, seed: int, pop_size: int = 20):
        super().__init__(seed)
        self.pop = space.sample_uniform(self.rng, pop_size)
        self.vel = self.VEL_COEFF * self.rng.standard_normal((pop_size, space.DIM))
        self.pbest = self.pop.copy()
        self.pbest_f = np.full(pop_size, np.inf)
        self.gbest: np.ndarray | None = None
        self.gbest_f = float("inf")

    def ask(self):
        self.pop = space.clip(self.pop)
        return [p.copy() for p in self.pop]

    def tell(self, thetas, fitnesses):
        self._update_best(thetas, fitnesses)
        f = np.asarray(fitnesses, dtype=float)
        improved = f < self.pbest_f
        self.pbest_f[improved] = f[improved]
        self.pbest[improved] = self.pop[improved]
        i_min = int(np.argmin(self.pbest_f))
        if self.pbest_f[i_min] < self.gbest_f:
            self.gbest_f = float(self.pbest_f[i_min])
            self.gbest = self.pbest[i_min].copy()
        for i in range(len(self.pop)):
            for j in range(space.DIM):
                r1 = self.rng.random(); r2 = self.rng.random(); r3 = self.rng.random()
                c1 = np.log10((self.LOG_MAX - self.LOG_MIN) * r1 + self.LOG_MIN)
                c2 = np.log10((self.LOG_MAX - self.LOG_MIN) * r2 + self.LOG_MIN)
                c3 = np.log10((self.POS_LOG_MAX - self.POS_LOG_MIN) * r3 + self.POS_LOG_MIN)
                self.vel[i, j] = (r1 * c1 * self.vel[i, j]
                                  + c2 * c3 * (self.pbest[i, j] - self.pop[i, j])
                                  + c2 * c3 * (self.gbest[j] - self.pop[i, j]))
        self.pop = self.pop + self.vel

class EnhancedSSO(SSO):
    """Enhanced Sperm Swarm Optimization (this work): standard SSO augmented with
    two established, citable improvements, applied only to the initial population
    so the per-iteration update stays a faithful SSO and the budget is spent the
    same way:
      (i)  Chaotic (logistic-map) initialization for a well-spread, diverse
           starting population (chaotic-map initialization is a standard
           metaheuristic enhancement for population diversity).
      (ii) Opposition-based learning (Rahnamayan et al.): each initial particle's
           opposite point is also evaluated and the better half is retained,
           giving a higher-quality start and faster convergence.
    After initialization the search reverts to the exact Shehadeh SSO update, so
    Enhanced SSO differs from SSO only through a superior, budget-counted start."""

    def __init__(self, seed: int, pop_size: int = 20):
        super().__init__(seed, pop_size)
        self.pop_size = pop_size
        self._init_phase = True
        # chaotic logistic-map initial population (x_{n+1} = 4 x_n (1 - x_n))
        x = self.rng.random(space.DIM)
        pts = []
        for _ in range(pop_size):
            x = 4.0 * x * (1.0 - x)
            pts.append(space.LOW + x * (space.HIGH - space.LOW))
        self.pop = np.array(pts)

    def ask(self):
        if self._init_phase:
            self.pop = space.clip(self.pop)
            opposite = space.LOW + space.HIGH - self.pop      # opposition-based learning
            self._cand = np.vstack([self.pop, opposite])
            return [c.copy() for c in self._cand]
        self.pop = space.clip(self.pop)
        return [p.copy() for p in self.pop]

    def tell(self, thetas, fitnesses):
        self._update_best(thetas, fitnesses)
        if self._init_phase:
            f = np.asarray(fitnesses, dtype=float)
            idx = np.argsort(f)[:self.pop_size]               # keep the best half
            self.pop = np.array(thetas)[idx].copy()
            self.vel = self.VEL_COEFF * self.rng.standard_normal((self.pop_size, space.DIM))
            self.pbest = self.pop.copy()
            self.pbest_f = f[idx].copy()
            i_min = int(np.argmin(self.pbest_f))
            self.gbest = self.pbest[i_min].copy()
            self.gbest_f = float(self.pbest_f[i_min])
            self._init_phase = False
            return
        super().tell(thetas, fitnesses)

class PSO(_Base):
    """Inertia-weight PSO: w=0.72, c1=c2=1.49, population 20 (Master Plan 6.2)."""
    W, C1, C2 = 0.72, 1.49, 1.49

    def __init__(self, seed: int, pop_size: int = 20):
        super().__init__(seed)
        self.pop = space.sample_uniform(self.rng, pop_size)
        span = (space.HIGH - space.LOW)
        self.vel = self.rng.uniform(-span, span, size=(pop_size, space.DIM)) * 0.1
        self.pbest = self.pop.copy()
        self.pbest_f = np.full(pop_size, np.inf)
        self.gbest: np.ndarray | None = None
        self.gbest_f = float("inf")

    def ask(self):
        self.pop = space.clip(self.pop)
        return [p.copy() for p in self.pop]

    def tell(self, thetas, fitnesses):
        self._update_best(thetas, fitnesses)
        f = np.asarray(fitnesses, dtype=float)
        improved = f < self.pbest_f
        self.pbest_f[improved] = f[improved]
        self.pbest[improved] = self.pop[improved]
        i_min = int(np.argmin(self.pbest_f))
        if self.pbest_f[i_min] < self.gbest_f:
            self.gbest_f = float(self.pbest_f[i_min])
            self.gbest = self.pbest[i_min].copy()
        r1 = self.rng.random((len(self.pop), space.DIM))
        r2 = self.rng.random((len(self.pop), space.DIM))
        self.vel = (self.W * self.vel
                    + self.C1 * r1 * (self.pbest - self.pop)
                    + self.C2 * r2 * (self.gbest[None, :] - self.pop))
        self.pop = self.pop + self.vel

class RandomSearch(_Base):
    """Uniform over bounds, same discretisation as everyone else."""
    def __init__(self, seed: int, batch: int = 20):
        super().__init__(seed)
        self.batch = batch

    def ask(self):
        return [t for t in space.sample_uniform(self.rng, self.batch)]

    def tell(self, thetas, fitnesses):
        self._update_best(thetas, fitnesses)

class RegularizedEvolution(_Base):
    """Population 20, tournament 5, one-dimension mutation (Real et al. 2019).
    First ask() returns the random initial population; afterwards children in
    batches of `batch` (default 1)."""
    def __init__(self, seed: int, pop_size: int = 20, tournament: int = 5, batch: int = 1):
        super().__init__(seed)
        self.pop_size, self.tournament, self.batch = pop_size, tournament, batch
        self.population: list[tuple[np.ndarray, float]] = []  # FIFO
        self._pending_init = space.sample_uniform(self.rng, pop_size)
        self._init_done = False

    def _mutate(self, parent: np.ndarray) -> np.ndarray:
        child = parent.copy()
        d = int(self.rng.integers(0, space.DIM))
        child[d] = self.rng.uniform(space.LOW[d], space.HIGH[d])
        return child

    def ask(self):
        if not self._init_done:
            return [t.copy() for t in self._pending_init]
        out = []
        for _ in range(self.batch):
            idx = self.rng.choice(len(self.population), size=min(self.tournament, len(self.population)), replace=False)
            parent = min((self.population[i] for i in idx), key=lambda tf: tf[1])[0]
            out.append(self._mutate(parent))
        return out

    def tell(self, thetas, fitnesses):
        self._update_best(thetas, fitnesses)
        for t, f in zip(thetas, fitnesses):
            self.population.append((np.asarray(t, dtype=float).copy(), float(f)))
            if self._init_done and len(self.population) > self.pop_size:
                self.population.pop(0)   # regularisation: remove OLDEST
        if not self._init_done:
            self._init_done = True
            while len(self.population) > self.pop_size:
                self.population.pop(0)

class TPE(_Base):
    """Optuna TPE on the identical space. Requires optuna (DICC env)."""
    def __init__(self, seed: int, batch: int = 1):
        super().__init__(seed)
        import optuna                                     # lazy: not needed locally
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._optuna = optuna
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed))
        self.batch = batch
        self._open: list = []

    def ask(self):
        self._open = []
        out = []
        for _ in range(self.batch):
            trial = self.study.ask()
            theta = np.empty(space.DIM)
            for d in range(space.DIM):
                if d in space.INT_DIMS:
                    theta[d] = trial.suggest_int(f"x{d}", int(space.LOW[d]), int(space.HIGH[d]))
                elif d in space.BIN_DIMS:
                    theta[d] = trial.suggest_int(f"x{d}", 0, 1)
                else:
                    theta[d] = trial.suggest_float(f"x{d}", float(space.LOW[d]), float(space.HIGH[d]))
            self._open.append(trial)
            out.append(theta)
        return out

    def tell(self, thetas, fitnesses):
        self._update_best(thetas, fitnesses)
        for trial, f in zip(self._open, fitnesses):
            self.study.tell(trial, float(f))
        self._open = []

STRATEGIES = {
    "sso": SSO,
    "esso": EnhancedSSO,
    "pso": PSO,
    "random": RandomSearch,
    "revol": RegularizedEvolution,
    "tpe": TPE,
}

def make(strategy: str, seed: int, **kw) -> Optimizer:
    return STRATEGIES[strategy](seed=seed, **kw)
