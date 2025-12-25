import json, os, ast

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Instance not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_instance(path):
    instance = load_json(path)

    if instance['gamma'] != 1:
        raise NotImplementedError("gamma=1 is expected.")

    instance['task_tr'] = {int(k):v for k, v in instance['task_tr'].items()}
    instance['r'] = {int(k):v for k, v in instance['r'].items()}
    instance['a'] = {int(k):v for k, v in instance['a'].items()}
    instance['b'] = {int(k):v for k, v in instance['b'].items()}
    instance['l^1'] = {int(k):v for k, v in instance['l^1'].items()}
    instance['l^2'] = {int(k):v for k, v in instance['l^2'].items()}
    instance['l^0'] = {int(k):v for k, v in instance['l^0'].items()}
    instance['g'] = {
        ast.literal_eval(k): v
        for k, v in instance['g'].items()
    }
    instance['V_tau'] = {int(k):v for k, v in instance['V_tau'].items()}
    instance['crane_tr'] = {int(k):v for k, v in instance['crane_tr'].items()}
    instance['es'] = {int(k):v for k, v in instance['es'].items()}
    instance['ls'] = {int(k):v for k, v in instance['ls'].items()}
    instance['d'] = {
        ast.literal_eval(k): v
        for k, v in instance['d'].items()
    }
    instance['Xi'] = set(tuple(pair) for pair in instance['Xi'])

    if 'h' in instance:
        instance['h'] = {int(k):v for k, v in instance['h'].items()}
    else:
        lamb = instance['lambda']
        gamma = instance['gamma']
        ls = instance['l^1']
        lt = instance['l^2']
        that = instance['hat(t)']
        T = instance['T']
        l0 = instance['l^0']
        V = instance['V']
        lmin = {t: min(ls[t], lt[t]) for t in T}
        lmax = {t: max(ls[t], lt[t]) for t in T}
        delta = {(v1,v2):(gamma + 1)*abs(v1-v2) for v1 in V for v2 in V}
        task_tr = instance['task_tr']
        Vt = instance['V_tau']

        instance['h'] = {t:abs(ls[t]-lt[t])*that+2*lamb for t in T}
    
    if 't^0' in instance:
        instance['t^0'] = {
            ast.literal_eval(k): v
            for k, v in instance['t^0'].items()
        }
    else:
        instance['t^0'] = {
            (v, t): abs(l0[v]-ls[t])*that
            for v in V
            for t in T
        }
    
    if 't' in instance:
        instance['t'] = {
            ast.literal_eval(k): v
            for k, v in instance['t'].items()
        }
    else:
        instance['t'] = {
            (t1, t2): abs(lt[t1]-ls[t2])*that
            for t1 in T
            for t2 in T
        }
    
    if 'Theta' in instance:
        instance['Theta'] = set(tuple(quad) for quad in instance['Theta'])
    else:
        instance['Theta'] = set(
            (t1, t2, v1, v2)
            for t1 in T
            for t2 in T
            for v1 in Vt[t1]
            for v2 in Vt[t2]
            if task_tr[t1] == task_tr[t2] and v1 != v2 and t1 != t2
            and ((v1 < v2 and lmax[t1] + delta[(v1,v2)] > lmin[t2])
                 or (v1 > v2 and lmin[t1] - delta[(v1,v2)] < lmax[t2]))
        )

    if 'Delta' in instance:
        instance['Delta'] = {
            ast.literal_eval(k): v
            for k, v in instance['Delta'].items()
        }
    else:
        instance['Delta'] = {}
        for t1, t2, v1, v2 in instance['Theta']:
            if v1 < v2:
                condition1 = lt[t1] + delta[(v1, v2)] > ls[t2]
                condition2 = ls[t1] + delta[(v1, v2)] > ls[t2]
                condition3 = lt[t1] + delta[(v1, v2)] > lt[t2]

                if condition1:
                    instance['Delta'][(t1, t2, v1, v2)] = (lt[t1] + delta[(v1, v2)] - ls[t2]) * that
                elif condition2 or condition3:
                    instance['Delta'][(t1, t2, v1, v2)] = (lt[t1] + delta[(v1, v2)] - ls[t2]) * that - lamb
                else:
                    instance['Delta'][(t1, t2, v1, v2)] = (lt[t1] + delta[(v1, v2)] - ls[t2]) * that - 2*lamb

            elif v1 > v2:
                condition1 = lt[t1] - delta[(v1, v2)] < ls[t2]
                condition2 = ls[t1] - delta[(v1, v2)] < ls[t2]
                condition3 = lt[t1] - delta[(v1, v2)] < lt[t2]

                if condition1:
                    instance['Delta'][(t1, t2, v1, v2)] = (ls[t2] + delta[(v1, v2)] - lt[t1]) * that
                elif condition2 or condition3:
                    instance['Delta'][(t1, t2, v1, v2)] = (ls[t2] + delta[(v1, v2)] - lt[t1]) * that - lamb
                else:
                    instance['Delta'][(t1, t2, v1, v2)] = (ls[t2] + delta[(v1, v2)] - lt[t1]) * that - 2*lamb
    print('Instance loaded:', path)
    return instance
    