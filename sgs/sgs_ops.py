from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Set, Tuple
import math

@dataclass(frozen=True)
class SGSState:
    S: Dict[int, float]          # start time of scheduled tasks
    ES: Dict[int, float]         # earliest start time
    W: Dict[int, Set[int]]       # successors
    Q: Dict[int, int]            # remaining predecessor count
    G: Dict[int, Set[int]]       # assigned tasks per crane
    C: Dict[int, float]          # crane available time
    L: Dict[int, int]            # crane position
    U: Set[int]                  # unscheduled tasks

def build_initial_state(instance: Dict[str, Any]) -> SGSState:
    T = instance["T"]
    es = instance["es"]
    Xi = instance["Xi"]
    V = instance["V"]
    l0 = instance["l^0"]

    S, ES, W, Q, G, C, L = {}, {}, {}, {}, {}, {}, {}

    Tset = set(T)

    for t in T:
        t = int(t)
        ES[t] = float(es[t])
        W[t] = set(t2 for t2 in Tset if (t, t2) in Xi)
        Q[t] = len(set(t2 for t2 in Tset if (t2, t) in Xi))

    for v in V:
        v = int(v)
        G[v] = set()
        C[v] = 0.0
        L[v] = int(l0[v])

    U = set(int(t) for t in T)

    return SGSState(S=S, ES=ES, W=W, Q=Q, G=G, C=C, L=L, U=U)

def ready_tasks(st: SGSState) -> list[int]:
    return [t for t in st.U if st.Q[t] == 0]

def compute_est(instance: Dict[str, Any], st: SGSState, t: int, v: int) -> float:
    """
    Mirrors your SGS-A time computation (without hard infeasible cut):
      est = max(ES[t], C[v] + |l1[t]-L[v]|*that, ... interference constraints ...)
    """
    l1 = instance["l^1"]
    that = instance["hat(t)"]
    Theta = instance["Theta"]
    h = instance["h"]
    Delta = instance["Delta"]
    Vt = instance["V_tau"][t]
    crane_tr = instance["crane_tr"]

    est = max(st.ES[t], st.C[v] + abs(l1[t] - st.L[v]) * that)

    R = []
    if est <= instance['ls'][t]:
        R.append((est, instance['ls'][t]))
        for v2 in Vt:
            if v2 == v:
                continue
            for t2 in st.G[v2]:
                if (t2, t, v2, v) in Theta:
                    Rprime = []
                    for (es_t, ls_t) in R:
                        new_ls = min(ls_t, st.S[t2] - Delta[(t, t2, v, v2)] - h[t])
                        if es_t <= new_ls:
                            Rprime.append((es_t, new_ls))
                        new_es = max(es_t, st.S[t2] + h[t2] + Delta[((t2, t, v2, v))])
                        if new_es <= instance['ls'][t]:
                            Rprime.append((new_es, ls_t))
                    R = Rprime
    if R:
        R.sort(key=lambda x:x[0])
        return float(R[0][0])

    # interference constraints: only compare cranes in same track
    for vprime in Vt:
        if vprime == v:
            continue
        for tprime in st.G[vprime]:
            # Your condition: if (tprime, tstar, vprime, v) in Theta
            key = (tprime, t, vprime, v)
            if key in Theta:
                # Delta indexed by (tprime, t, vprime, v)
                est = max(est, st.S[tprime] + h[tprime] + Delta[key])

    return float(est)

def apply_action(instance: Dict[str, Any], st: SGSState, t: int, v: int, est: float) -> SGSState:
    """
    Apply chosen (t,v) with computed est. Mirrors SGS-A updates.
    """
    h = instance["h"]
    l2 = instance["l^2"]
    g = instance["g"]

    # copy mutables (simple/debug-friendly; optimize later)
    S = dict(st.S)
    ES = dict(st.ES)
    W = {k: set(vv) for k, vv in st.W.items()}
    Q = dict(st.Q)
    G = {k: set(vv) for k, vv in st.G.items()}
    C = dict(st.C)
    L = dict(st.L)
    U = set(st.U)

    S[t] = est
    G[v].add(t)
    C[v] = est + h[t]
    L[v] = l2[t]
    U.remove(t)

    # successor updates
    for j in W[t]:
        Q[j] -= 1
        ES[j] = max(ES[j], S[t] + g[(t, j)])

    return SGSState(S=S, ES=ES, W=W, Q=Q, G=G, C=C, L=L, U=U), est

def compute_obj_delay(instance: Dict[str, Any], st: SGSState) -> tuple[float, float]:
    """
    Same as your SGS-A:
      obj = sum(S[t] + h[t] - r[t] for t in Tobj)
      delay = sum(max(0, S[t] - b[t]) for t in T)
    """
    T = instance["T"]
    Tobj = instance["T^obj"]
    h = instance["h"]
    r = instance["r"]
    b = instance["b"]

    obj = sum(st.S[t] + h[t] - r[t] for t in Tobj)
    delay = sum(max(0.0, st.S[int(t)] - b[int(t)]) for t in T)
    return float(obj), float(delay)
