import random

def sgs_i(instance):

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
    ls = instance['ls']

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




    traj = []
    while U:
        A = list(t for t in U if Q[t] == 0)
        A.sort(key=lambda x:(ES[x],x))
        t1 = A[0]

        Vprime = list()
        E = dict()

        if t1 == 45:
            pass


        for v1 in Vt[t1]:
            est = max(ES[t1], C[v1] + abs(l1[t1] - L[v1]) * that)
            if est > ls[t1]:
                continue

            R = []
            R.append((est,ls[t1]))

            for v2 in Vt[t1]:
                if v1 == v2:
                    continue
                
                for t2 in G[v2]:
                    if (t1,t2,v1,v2) in Theta:
                        Rprime = []

                        for (es_t1, ls_t1) in R:
                            new_ls = min(ls_t1, S[t2] - Delta[t1,t2,v1,v2] - h[t1])
                            if es_t1 <= new_ls:
                                Rprime.append((es_t1, new_ls))
                            
                            new_es = max(es_t1, S[t2] + h[t2] + Delta[t2,t1,v2,v1])
                            if new_es <= ls_t1:
                                Rprime.append((new_es, ls_t1))
                        
                        R = Rprime
                
            if R:
                Vprime.append(v1)
                R.sort(key=lambda x:x[0])
                E[v1] = R[0][0]

        Vprime.sort(key=lambda x:(E[x],x))
        vstar = Vprime[0]
        tstar = t1
        S[tstar] = E[vstar]
        G[vstar].add(tstar)
        C[vstar] = E[vstar] + h[tstar]
        L[vstar] = l2[tstar]
        U.remove(tstar)
        traj.append((tstar, vstar, S[tstar]))

        for t in W[tstar]:
            Q[t] -= 1
            ES[t] = max(ES[t], S[tstar] + g[tstar,t])

    obj = sum(S[t] + h[t] - r[t] for t in Tobj)

    delay = sum(max(0, S[t] - b[t]) for t in T)
    # print(traj)
    return S, G, obj, delay


if __name__ == "__main__":
    import time
    from tqdm import tqdm
    from utils import load_instance
    
    def main(n_cranes, n_jobs, instance_idx):
        inst_path = f"./instances/{n_cranes}_{n_jobs}_{instance_idx}.json"

        start = time.time()
        instance = load_instance(inst_path)
        load_time = time.time() - start
        print(f"[INFO] Instance loaded in {load_time:.4f} sec")

        results = []
        start = time.time()
        schedule, assignment, obj, delay = sgs_i(instance)
        runtime = time.time() - start
        print(f"[INFO] Heuristic executed in {runtime:.4f} sec")

        results.append((obj, delay))
        results.sort(key=lambda x: (x[1], x[0]))
        print(instance_idx, results[0])
    
    for idx in range(30):
        main(
            n_cranes=3,
            n_jobs=50,
            instance_idx=idx
        )