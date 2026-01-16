import random

def sgs_a(instance):

    T = instance['T']
    a = instance['a']
    b = instance['b']
    Xi = instance['Xi']
    V = instance['V']
    l0 = instance['l^0']
    Vt = instance['V_tau']
    l1 = instance['l^1']
    l2 = instance['l^2']
    that = instance['hat(t)']
    Theta = instance['Theta']
    h = instance['h']
    Delta = instance['Delta']
    g = instance['g']
    Tobj = instance['T^obj']
    r = instance['r']
    crane_tr = instance['crane_tr']

    S, ES, W, Q, G, C, L = {}, {}, {}, {}, {}, {}, {}

    for t in T:
        ES[t] = a[t]
        W[t] = set(t2 for t2 in T if (t, t2) in Xi)
        Q[t] = len(set(t2 for t2 in T if (t2, t) in Xi))
    
    for v in V:
        G[v] = set()
        C[v] = 0
        L[v] = l0[v]

    U = set(T)

    while U:
        A = set(t for t in U if Q[t] == 0)
        tstar = min(A, key=lambda t: ES[t])


        # min_val = min(ES[t] for t in A)
        # tstars = [t for t in A if ES[t] == min_val]
        # if len(tstars) > 1:
        #     pass
        # tstar = random.choice(tstars)


        Vprime = set()
        E = dict()

        for v in Vt[tstar]:
            est = max(ES[tstar], C[v] + abs(l1[tstar] - L[v]) * that)
            for vprime in V:
                if crane_tr[vprime] != crane_tr[v]:
                    continue
                if vprime == v:
                    continue
                for tprime in G[vprime]:
                    if (tprime, tstar, vprime, v) in Theta:
                        est = max(est, S[tprime] + h[tprime] + Delta[(tprime, tstar, vprime, v)])
                
                Vprime.add(v)
                E[v] = est

        vstar = min(Vprime, key=lambda v: E[v])


        # min_val = min(E[v] for v in Vprime)
        # vstars = [v for v in Vprime if E[v] == min_val]
        # if len(vstars) > 1:
        #     pass
        # vstar = random.choice(vstars)


        S[tstar] = E[vstar]
        G[vstar].add(tstar)
        C[vstar] = E[vstar] + h[tstar]
        L[vstar] = l2[tstar]
        U.remove(tstar)

        for t in W[tstar]:
            Q[t] -= 1
            ES[t] = max(ES[t], S[tstar] + g[tstar,t])

    obj = sum(S[t] + h[t] - r[t] for t in Tobj)

    delay = sum(max(0, S[t] - b[t]) for t in T)

    return S, G, obj, delay