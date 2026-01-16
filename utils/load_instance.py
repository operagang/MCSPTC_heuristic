import json, os, ast

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Instance not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_instance(path, apply_rules=True):
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
    
    instance['successors'] = {t:[succ for (pred,succ) in instance['Xi'] if pred == t] for t in instance['T']}

    if apply_rules:
        set_prec_dist(instance)
        rule_1_added = rule_1(instance)
        set_dist_matrix(instance)
        set_start_window(instance)
        set_prec_dist(instance)

        rule_2_added = rule_2(instance)
        set_dist_matrix(instance)
        set_start_window(instance)
        set_prec_dist(instance)

        rule_3_added = rule_3(instance)

    pass









    print('Instance loaded:', path)
    return instance



def rule_1(inst):
    tasks = inst['T']
    es = inst['es']
    ls = inst['ls']
    prec_dist = inst['prec_dist']
    task_tr = inst['task_tr']
    h = inst['h']
    travel = inst['t']
    V_tau = inst['V_tau']
    Theta = inst['Theta']
    Delta = inst['Delta']
    successors = inst['successors']
    g = inst['g']
    Xi = inst['Xi']

    new_prec = []
    for t1 in tasks:
        t1_ls = ls[t1]
        for t2 in tasks:
            if t1 != t2:
                if prec_dist[t1][t2] == float('-inf') and prec_dist[t2][t1] == float('-inf'):
                    if task_tr[t1] == task_tr[t2]:
                        t2_ef = es[t2] + h[t2]
                        min_td = travel[t2, t1]
                        for v1 in V_tau[t1]:
                            for v2 in V_tau[t2]:
                                if v1 == v2:
                                    continue
                                if (t2, t1, v2, v1) in Theta:
                                    min_td = min(min_td, Delta[t2, t1, v2, v1])
                                else:
                                    min_td = float('-inf')
                                    break
                            if min_td == float('-inf'):
                                break

                        if t2_ef + min_td > t1_ls:
                            new_prec.append((t1, t2))

    num_added = 0
    for t1, t2 in new_prec:
        if not dipath_exists_acyclic(successors, t2, t1):
            if t2 not in successors[t1]:
                successors[t1].append(t2)
                num_added += 1
            lag = es[t2] - ls[t1]
            if (t1,t2) in g:
                g[t1,t2] = max(g[t1,t2], lag)
            else:
                g[t1,t2] = lag
            if (t1,t2) not in Xi:
                Xi.add((t1,t2))
    
    return num_added


def rule_2(inst):
    tasks = inst['T']
    es = inst['es']
    ls = inst['ls']
    h = inst['h']
    prec_dist = inst['prec_dist']
    task_tr = inst['task_tr']
    Theta = inst['Theta']
    Delta = inst['Delta']
    travel = inst['t']
    V_tau = inst['V_tau']
    successors = inst['successors']
    g = inst['g']
    Xi = inst['Xi']

    rail_pred_list = {}
    rail_succ_list = {}

    for t in tasks:
        rail_pred_list[t] = []
        rail_succ_list[t] = []

    for t_succ in tasks:
        for t_pred in tasks:
            if t_pred != t_succ:
                if task_tr[t_pred] == task_tr[t_succ]:
                    if prec_dist[t_pred][t_succ] > float('-inf'):
                        rail_pred_list[t_succ].append(t_pred)
                        rail_succ_list[t_pred].append(t_succ)

    new_prec = []
    for t1 in tasks:
        t1_lf = ls[t1] + h[t1]
        for t2 in tasks:
            if t1 == t2:
                continue
            if prec_dist[t1][t2] == float('-inf') and prec_dist[t2][t1] == float('-inf'):
                if task_tr[t1] != task_tr[t2]:
                    continue
                t2_es = es[t2]

                lag = travel[t1,t2]
                for v1 in V_tau[t1]:
                    for v2 in V_tau[t2]:
                        if v1 == v2:
                            continue
                        if (t1, t2, v1, v2) in Theta:
                            lag = max(lag, Delta[t1, t2, v1, v2])
                
                if t1_lf + lag <= t2_es:
                    add_t1_to_t2 = True

                    for t2_succ in rail_succ_list[t2]:
                        if prec_dist[t1][t2_succ] == float('-inf'):
                            t2_succ_es = es[t2_succ]

                            lag = travel[t1,t2_succ]
                            for v1 in V_tau[t1]:
                                for v2 in V_tau[t2_succ]:
                                    if v1 == v2:
                                        continue
                                    if (t1, t2_succ, v1, v2) in Theta:
                                        lag = max(lag, Delta[t1, t2_succ, v1, v2])
                            if t1_lf + lag > t2_succ_es:
                                add_t1_to_t2 = False
                                break

                    if not add_t1_to_t2:
                        continue
                    
                    for t1_pred in rail_pred_list[t1]:
                        if prec_dist[t1_pred][t2] == float('-inf'):
                            t1_pred_lf = ls[t1_pred] + h[t1_pred]

                            lag = travel[t1_pred,t2]
                            for v1 in V_tau[t1_pred]:
                                for v2 in V_tau[t2]:
                                    if v1 == v2:
                                        continue
                                    if (t1_pred, t2, v1, v2) in Theta:
                                        lag = max(lag, Delta[t1_pred, t2, v1, v2])
                            if t1_pred_lf + lag > t2_es:
                                add_t1_to_t2 = False
                                break
                            
                            for t2_succ in rail_succ_list[t2]:
                                if prec_dist[t1_pred][t2_succ] == float('-inf'):
                                    t2_succ_es = es[t2_succ]

                                    lag = travel[t1_pred,t2_succ]
                                    for v1 in V_tau[t1_pred]:
                                        for v2 in V_tau[t2_succ]:
                                            if v1 == v2:
                                                continue
                                            if (t1_pred, t2_succ, v1, v2) in Theta:
                                                lag = max(lag, Delta[t1_pred, t2_succ, v1, v2])
                                    if t1_pred_lf + lag > t2_succ_es:
                                        add_t1_to_t2 = False
                                        break
                        
                        if not add_t1_to_t2:
                            break
                    
                    if add_t1_to_t2:
                        new_prec.append((t1,t2))

    num_added = 0
    for t1,t2 in new_prec:
        if not dipath_exists_acyclic(successors, t2, t1):
            if t2 not in successors[t1]:
                successors[t1].append(t2)
                num_added += 1
            lag = es[t2] - ls[t1]
            if (t1,t2) in g:
                g[t1,t2] = max(g[t1,t2], lag)
            else:
                g[t1,t2] = lag
            if (t1,t2) not in Xi:
                Xi.add((t1,t2))
    
    return num_added


def rule_3(inst):
    tasks = inst['T']
    es = inst['es']
    ls = inst['ls']
    prec_dist = inst['prec_dist']
    task_tr = inst['task_tr']
    successors = inst['successors']
    g = inst['g']
    Xi = inst['Xi']

    no_prec = []
    for t1 in tasks:
        for t2 in tasks:
            if t1 < t2:
                if task_tr[t1] != task_tr[t2]:
                    if (prec_dist[t1][t2] == float('-inf')) and (prec_dist[t2][t1] == float('-inf')):
                        no_prec.append((t1, t2))
    
    conflict_dict = {} # conflict_dict 는 new prec 추가해도 불변
    for t in tasks:
        conflict_dict[t] = []
        for other_t in tasks:
            if other_t != t and task_tr[t] == task_tr[other_t]:
                if prec_dist[t][other_t] == float('-inf') and prec_dist[other_t][t] == float('-inf'):
                    conflict_dict[t].append(other_t)
    
    possible_dict = {}
    for t1 in prec_dist:
        possible_dict[t1] = {}
        for t2 in prec_dist[t1]:
            if t1 != t2:
                if prec_dist[t1][t2] > float('-inf'):
                    possible_dict[t1][t2] = True
                else:
                    possible_dict[t1][t2] = False
    for t1 in conflict_dict:
        for t2 in conflict_dict[t1]:
            possible_dict[t1][t2] = True # 반대방향도 고려됨
    
    possible_mat = {}
    for t1 in possible_dict:
        possible_mat[t1] = []
        for t2 in possible_dict[t1]:
            if possible_dict[t1][t2]:
                possible_mat[t1].append(t2)

    num_added = 0
    while len(no_prec) != 0:
        prec = no_prec.pop()

        # cycle도 방지됨
        if (prec_dist[prec[0]][prec[1]] != float('-inf')) or (prec_dist[prec[1]][prec[0]] != float('-inf')):
            continue

        t1, t2 = prec
        new_prec = None

        es_t1 = es[t1]
        es_t2 = es[t2]
        if es_t1 < es_t2:
            if not dipath_exists_cyclic(possible_mat, t2, t1):
                new_prec = (t1, t2)
            elif not dipath_exists_cyclic(possible_mat, t1, t2):
                new_prec = (t2, t1)
            else:
                continue
        else:
            if not dipath_exists_cyclic(possible_mat, t1, t2):
                new_prec = (t2, t1)
            elif not dipath_exists_cyclic(possible_mat, t2, t1):
                new_prec = (t1, t2)
            else:
                continue
        
        t_pred = new_prec[0]
        t_succ = new_prec[1]

        if (t_pred, t_succ) not in Xi:
            Xi.add((t_pred,t_succ))
            successors[t_pred].append(t_succ)
            num_added += 1
        else:
            sys.exit('prec 추가중 error')

        lag = es[t_succ] - ls[t_pred]
        if (t_pred,t_succ) not in g:
            g[t_pred,t_succ] = lag
        else:
            sys.exit('prec 추가중 error')
        
        possible_mat[t_pred].append(t_succ)

        prec_dist[t_pred][t_succ] = 1
        for t1 in tasks:
            for t2 in tasks:
                if prec_dist[t1][t2] < prec_dist[t1][t_pred] + prec_dist[t_pred][t_succ] + prec_dist[t_succ][t2]:
                    prec_dist[t1][t2] = prec_dist[t1][t_pred] + prec_dist[t_pred][t_succ] + prec_dist[t_succ][t2]

    return num_added


def dipath_exists_cyclic(possible_mat, t1, t2):
    active_nodes = [t1]
    added_tf = {t:False for t in possible_mat}
    added_tf[t1] = True

    while len(active_nodes) > 0:
        node = active_nodes.pop()
        for next_node in possible_mat[node]:
            if next_node == t2:
                return True
            if not added_tf[next_node]:
                active_nodes.append(next_node)
                added_tf[next_node] = True

    return False

def dipath_exists_acyclic(succ_list, t1, t2):
    active_nodes = [t1]
    added_tf = {t:False for t in succ_list}
    added_tf[t1] = True

    while len(active_nodes) > 0:
        node = active_nodes.pop()
        for next_node in succ_list[node]:
            if next_node == t2:
                return True
            if not added_tf[next_node]:
                active_nodes.append(next_node)
                added_tf[next_node] = True
    
    return False


def set_start_window(inst):
    for t in inst['T']:
        inst['es'][t] = max(inst['es'][t], inst['d'][-1,t])
        inst['ls'][t] = min(inst['ls'][t], -inst['d'][t,-1])

import sys
def set_dist_matrix(inst):
    tasks = inst['T']
    es = inst['es']
    ls = inst['ls']
    succ_list = inst['successors']
    g = inst['g']

    dist = {}

    for t in tasks+[-1]:
        for t in tasks+[-1]:
            dist[t,t] = {}

    dist[-1,-1] = 0
    for t in tasks:
        dist[-1,t] = es[t]
        dist[t,-1] = -ls[t]

    for t1 in tasks:
        for t2 in tasks:
            if t1 == t2:
                dist[t1,t2] = 0
            else:
                if t2 in succ_list[t1]:
                    dist[t1,t2] = g[t1, t2]
                else:
                    dist[t1,t2] = float('-inf')
    for t in tasks+[-1]:
        for k in tasks+[-1]:
            for l in tasks+[-1]:
                if t != k and t != l:
                    if dist[k,t] + dist[t,l] > dist[k,l]:
                        dist[k,l] = dist[k,t] + dist[t,l]

    for t in tasks+[-1]:
        if dist[t,t] > 0:
            sys.exit(f'Pos Cycle exists: {t}')
    
    inst['d'] = dist
    pass


def set_prec_dist(inst):
    tasks = inst['T']
    Xi = inst['Xi']
    succ_list = inst['successors']

    dist = {}

    for t in tasks:
        dist[t] = {}
    
    for t1 in tasks:
        for t2 in tasks:
            if t1 == t2:
                dist[t1][t2] = 0
            else:
                if t2 in succ_list[t1]:
                    dist[t1][t2] = 1
                else:
                    dist[t1][t2] = float('-inf')

    for t in tasks:
        for k in tasks:
            for l in tasks:
                if t != k and t != l:
                    if dist[k][t] + dist[t][l] > dist[k][l]:
                        dist[k][l] = dist[k][t] + dist[t][l]
    
    inst['prec_dist'] = dist
    
    pass