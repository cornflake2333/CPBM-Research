"""J-K local energy per site, evaluated on a projected sampling state."""
import numpy as np

def local_energy(state):
    """Return complex e_J,e_K per site plus integer Q_nn and S(pi,pi)."""
    counts = state.occ[state.neighbors].sum(axis=1)
    a, b = state.edges.T
    active = state.occ[a] != state.occ[b]
    old = np.where(state.occ[a[active]], a[active], b[active])
    new = np.where(state.occ[a[active]], b[active], a[active])
    rows = state.locator[old]
    ratio = np.ones(len(old), complex)
    for alpha in range(2):
        ratio *= np.einsum('ij,ji->i', state.U[alpha, new], state.inverse[alpha][:, rows])
    ratio *= np.exp(np.clip(-state.g * (counts[new] - counts[old] - 1) + state.h * (state.stagger[new] - state.stagger[old]), -700, 700))
    ej = -ratio.sum() / state.N
    p = state.plaquettes
    n = state.occ[p]
    active = (n[:, 0] == n[:, 2]) & (n[:, 1] == n[:, 3]) & (n[:, 0] != n[:, 1])
    p = p[active]
    n = n[active]
    old0 = np.where(n[:, 0], p[:, 0], p[:, 1])
    old1 = np.where(n[:, 0], p[:, 2], p[:, 3])
    new0 = np.where(n[:, 0], p[:, 1], p[:, 0])
    new1 = np.where(n[:, 0], p[:, 3], p[:, 2])
    row0 = state.locator[old0]
    row1 = state.locator[old1]
    ring = np.ones(len(p), complex)
    for alpha in range(2):
        u0, u1 = (state.U[alpha, new0], state.U[alpha, new1])
        c0, c1 = (state.inverse[alpha][:, row0], state.inverse[alpha][:, row1])
        aa = np.einsum('ij,ji->i', u0, c0)
        ab = np.einsum('ij,ji->i', u0, c1)
        ba = np.einsum('ij,ji->i', u1, c0)
        bb = np.einsum('ij,ji->i', u1, c1)
        ring *= aa * bb - ab * ba
    delta = counts[new0] + counts[new1] - counts[old0] - counts[old1] - 4
    dm = state.stagger[new0] + state.stagger[new1] - state.stagger[old0] - state.stagger[old1]
    ring *= np.exp(np.clip(-state.g * delta + state.h * dm, -700, 700))
    ek = ring.sum() / state.N
    q = state.q_nn()
    spi = float(state.stagger @ state.occ) ** 2 / state.N
    return (complex(ej), complex(ek), q, spi)
