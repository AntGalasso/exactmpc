import numpy as np
from itertools import combinations
from math import comb
from exactmpc.exact import ExactQP


def single_n(P, q, G, h, tol=1e-9, rcond=1e-12):
    P = 0.5 * (P + P.T)
    Pinv = np.linalg.inv(P)
    k = G.shape[0]
    d = np.linalg.norm(G, axis=1)
    d[d == 0] = 1.0
    Gn = G / d[:, None]
    T = Gn @ Pinv @ Gn.T
    ustar = -Pinv @ q
    r0 = Gn @ ustar - h / d
    total = sum(comb(k, s) for s in range(1, min(k, P.shape[0]) + 1))
    visited = 0
    if np.all(r0 <= tol):
        return ustar, np.zeros(k), (), 0, total
    incumbent = np.inf
    dead = []
    best = None
    for size in range(1, min(k, P.shape[0]) + 1):
        for J in combinations(range(k), size):
            Jset = set(J)
            if any(D <= Jset for D in dead):
                continue
            Jl = list(J)
            S = T[np.ix_(Jl, Jl)]
            dg = np.sqrt(np.diag(S))
            lam = np.linalg.eigvalsh(S / np.outer(dg, dg))
            visited += 1
            if lam[0] <= rcond:
                dead.append(Jset)
                continue
            rJ = r0[Jl]
            nu = np.linalg.solve(S, rJ)
            price = 0.5 * rJ @ nu
            if np.all(r0 - T[:, Jl] @ nu <= tol):
                incumbent = min(incumbent, price)
                if np.all(nu >= -tol):
                    best = (Jl, nu, price)
            if price > incumbent + tol:
                dead.append(Jset)
    if best is None:
        return None, None, None, visited, total
    Jl, nu, _ = best
    nu_norm = np.zeros(k)
    nu_norm[Jl] = nu
    nuf = np.zeros(k)
    nuf[Jl] = nu / d[Jl]
    return ustar - Pinv @ (Gn.T @ nu_norm), nuf, tuple(Jl), visited, total


def batch_n(P, q, G, h, tol=1e-9, rcond=1e-12):
    P = 0.5 * (P + P.T)
    q = np.asarray(q, float)
    h = np.asarray(h, float)
    if q.ndim == 1:
        q = q[:, None]
    if h.ndim == 1:
        h = h[:, None]
    B = max(q.shape[1], h.shape[1])
    q = np.repeat(q, B, axis=1) if q.shape[1] == 1 else q
    h = np.repeat(h, B, axis=1) if h.shape[1] == 1 else h
    Pinv = np.linalg.inv(P)
    k = G.shape[0]
    d = np.linalg.norm(G, axis=1)
    d[d == 0] = 1.0
    Gn = G / d[:, None]
    T = Gn @ Pinv @ Gn.T
    ustar = -Pinv @ q
    r0 = Gn @ ustar - h / d[:, None]
    total = sum(comb(k, s) for s in range(1, min(k, P.shape[0]) + 1))
    visited = np.zeros(B, int)
    best = np.full(B, np.inf)
    nu_best = np.zeros((k, B))
    found = np.zeros(B, bool)
    dead = [[] for _ in range(B)]
    ok = np.all(r0 <= tol, axis=0)
    found |= ok
    alive_glob = ~ok
    for size in range(1, min(k, P.shape[0]) + 1):
        for J in combinations(range(k), size):
            Jmask = sum(1 << j for j in J)
            alive = np.array(
                [alive_glob[b] and not any(D & ~Jmask == 0 for D in dead[b]) for b in range(B)]
            )
            if not alive.any():
                continue
            visited[alive] += 1
            Jl = list(J)
            S = T[np.ix_(Jl, Jl)]
            dg = np.sqrt(np.diag(S))
            lam = np.linalg.eigvalsh(S / np.outer(dg, dg))
            if lam[0] <= rcond:
                for b in np.where(alive)[0]:
                    dead[b].append(Jmask)
                continue
            idx = np.where(alive)[0]
            rJ = r0[np.ix_(Jl, idx)]
            nuJ = np.linalg.solve(S, rJ)
            price = 0.5 * np.sum(rJ * nuJ, axis=0)
            viol = r0[:, idx] - T[:, Jl] @ nuJ
            primal = np.all(viol <= tol, axis=0)
            for m, b in enumerate(idx):
                if primal[m]:
                    best[b] = min(best[b], price[m])
                    if np.all(nuJ[:, m] >= -tol):
                        nu_best[:, b] = 0.0
                        nu_best[Jl, b] = nuJ[:, m] / d[Jl]
                        found[b] = True
                if price[m] > best[b] + tol:
                    dead[b].append(Jmask)
    nu_norm_best = np.zeros((k, B))
    for m, b in enumerate(np.where(found)[0]):
        nu_norm_best[:, b] = nu_best[:, b] * d
    u = ustar - Pinv @ Gn.T @ nu_norm_best
    u[:, ~found] = np.nan
    nu_best[:, ~found] = np.nan
    return u, nu_best, np.nan_to_num(nu_best) > 0, visited, np.full(B, total, int)


def rq(rng, n=8, k=5):
    X = rng.standard_normal((20, n))
    P = 2 * X.T @ X + 1e-2 * np.eye(n)
    return (
        P,
        rng.standard_normal(n) * 3,
        rng.standard_normal((k, n)),
        rng.standard_normal(k) * 0.5 + 0.5,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    mis = nanb = 0
    maxp = 0.0
    for _ in range(200):
        P, q, G, h = rq(rng)
        res = ExactQP(P, G).solve(q, h)
        b = batch_n(P, q, G, h)
        if res.found[0]:
            if np.isnan(b[0][:, 0]).any():
                nanb += 1
            else:
                mis += int(not np.allclose(b[0][:, 0], res.u[:, 0], atol=1e-8, rtol=0))
                r0 = G @ (-np.linalg.solve(P, q)) - h
                maxp = max(maxp, abs(0.5 * r0 @ b[1][:, 0] - res.price[0]))
    print("1) 200 QP: mismatch", mis, "nan", nanb, "max|price|", maxp)

    bad = tot = 0
    for _ in range(100):
        n, k, B = 5, 6, 6
        A = rng.normal(size=(n, n))
        P = A @ A.T + np.eye(n) * 0.2
        G = rng.normal(size=(k, n))
        qs = rng.normal(size=(n, B))
        us = rng.normal(size=(n, B))
        hs = np.einsum("kn,nb->kb", G, us) - rng.uniform(0, 1, size=(k, B))
        _, _, _, vis, _ = batch_n(P, qs, G, hs)
        for b in range(B):
            tot += 1
            bad += vis[b] != single_n(P, qs[:, b], G, hs[:, b])[3]
    print("2) visited parity fails", bad, "/", tot)

    P = np.eye(3)
    G = np.eye(3)
    q = np.zeros(3)
    h = np.ones(3)
    print("3) empty-set single", single_n(P, q, G, h)[3], "batch", batch_n(P, q, G, h)[3], "total", single_n(P, q, G, h)[4])

    G6 = rng.normal(size=(6, 3))
    q3 = rng.normal(size=3)
    h6 = rng.normal(size=6)
    s = single_n(np.eye(3), q3, G6, h6)
    bb = batch_n(np.eye(3), q3, G6, h6)
    print("4) k=6,n=3 total", s[4], bb[4][0], "expected", sum(comb(6, x) for x in (1, 2, 3)),
          "visited<=total", s[3] <= s[4], bb[3][0] <= bb[4][0])

    maxd = 0.0
    badn = cnt = 0
    for _ in range(200):
        n = 4
        A = rng.normal(size=(n, n))
        P = A @ A.T + np.eye(n) * 0.1
        G = rng.normal(size=(4, n)) * np.array([1, 1e-6, 1, 1])[:, None]
        q = rng.normal(size=n)
        uf = rng.normal(size=n)
        h = G @ uf - rng.uniform(0, 0.3, 4)
        res = ExactQP(P, G).solve(q, h)
        b = batch_n(P, q, G, h)
        if not res.found[0]:
            continue
        cnt += 1
        if np.isnan(b[0][:, 0]).any():
            badn += 1
        else:
            maxd = max(maxd, np.max(np.abs(b[0][:, 0] - res.u[:, 0])))
    print("7) scaling found-both", cnt, "nan", badn, "max|du|", maxd)

    mis2 = nan2 = 0
    for _ in range(30):
        n = 5
        A = rng.normal(size=(n, n))
        P = A @ A.T + np.eye(n) * 0.2
        G = rng.normal(size=(5, n))
        G[3] = G[1].copy()
        q = rng.normal(size=n)
        uf = rng.normal(size=n)
        h = G @ uf - rng.uniform(0, 0.3, 5)
        h[3] = h[1]
        res = ExactQP(P, G).solve(q, h)
        b = batch_n(P, q, G, h)
        if res.found[0]:
            if np.isnan(b[0][:, 0]).any():
                nan2 += 1
            else:
                mis2 += int(not np.allclose(b[0][:, 0], res.u[:, 0], atol=1e-8, rtol=0))
    print("6) dependent rows mismatch", mis2, "nan", nan2, "/30")
