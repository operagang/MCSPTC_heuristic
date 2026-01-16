from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any
import math
import random

from sgs.sgs_ops import (
    SGSState, build_initial_state, ready_tasks, compute_est, apply_action, compute_obj_delay
)

# action: (task, crane)
Action = Tuple[int, int]

@dataclass
class Node:
    state: SGSState
    parent: Optional["Node"] = None
    parent_action: Optional[Action] = None

    children: Dict[Action, "Node"] = field(default_factory=dict)
    untried_actions: List[Action] = field(default_factory=list)

    N: int = 0
    W: float = -float("inf")
    Q: float = 0.0  # mean reward

    expended_est: Optional[float] = None

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def uct_select_child(self, c_uct: float) -> "Node":
        assert self.children
        logN = math.log(self.N) if self.N > 0 else 0.0
        best, best_score = None, -float("inf")
        for a, ch in self.children.items():
            if ch.N == 0:
                score = float("inf")
            else:
                score = ch.Q + c_uct * math.sqrt(logN / ch.N)
            if score > best_score:
                best, best_score = ch, score
            # print(score)
        return best


class MCTS:
    def __init__(
        self,
        instance: Dict[str, Any],
        refs: Dict[str, float],
        seed: int = 0,
        c_uct: float = 1.4,
        n_rollout_limit: Optional[int] = None,
        reward_fail: float = -2.5,
        final_action: str = "visits",  # "visits" or "q"
    ):
        self.instance = instance
        self.refs = refs
        self.rng = random.Random(seed)
        self.c_uct = c_uct
        self.n_rollout_limit = n_rollout_limit
        self.reward_fail = reward_fail
        self.final_action = final_action

    # ---------- action generation ----------
    def legal_actions(self, st: SGSState) -> List[Action]:
        """
        Actions from current SGS state:
          choose any ready task t, then any feasible crane v in V_tau[t]
        No hard deadline filter here; delay is handled in reward.
        """
        Vt = self.instance["V_tau"]
        A = ready_tasks(st)
        actions: List[Action] = []
        for t in A:
            for v in Vt[t]:
                actions.append((t, v))
        return actions

    def step(self, st: SGSState, action: Action, est=None) -> SGSState:
        t, v = action
        if est is None:
            est = compute_est(self.instance, st, t, v)
        return apply_action(self.instance, st, t, v, est)

    # ---------- MCTS loop ----------
    def search(self, root_state: SGSState, n_iter: int) -> Tuple[Optional[Action], Node]:
        root = Node(state=root_state)
        root.untried_actions = self.legal_actions(root_state)
        self.rng.shuffle(root.untried_actions)

        for _ in range(n_iter):
            leaf = self._select(root)
            expanded = self._expand2(leaf)
            reward = self._simulate_sgs_i(expanded.state)
            self._backprop2(expanded, reward)

        return self._choose_final_action(root)

    def _select(self, node: Node) -> Node:
        # walk down by UCT while fully expanded
        while node.state.U and node.is_fully_expanded() and node.children:
            node = node.uct_select_child(self.c_uct)
        return node

    def _expand(self, node: Node) -> Node:
        if not node.state.U:
            return node
        if not node.untried_actions:
            return node

        action = node.untried_actions.pop()
        nxt, est = self.step(node.state, action)

        child = Node(state=nxt, parent=node, parent_action=action)
        child.untried_actions = self.legal_actions(nxt)
        self.rng.shuffle(child.untried_actions)
        child.expended_est = est

        node.children[action] = child
        return child
    
    def _expand2(self, node: Node) -> Node:
        if not node.state.U:
            return node
        if not node.untried_actions:
            return node

        # action = node.untried_actions.pop()
        node.untried_actions.sort(key=lambda x:(node.state.ES[x[0]], x[0], x[1]), reverse=True)
        action = node.untried_actions.pop()
        nxt, est = self.step(node.state, action)
        if action == (10,3):
            pass

        child = Node(state=nxt, parent=node, parent_action=action)
        child.untried_actions = self.legal_actions(nxt)
        self.rng.shuffle(child.untried_actions)
        child.expended_est = est

        node.children[action] = child
        return child

    def _simulate(self, st: SGSState) -> float:
        """
        Rollout policy: RANDOM task + RANDOM crane, then compute time like SGS-A.
        """
        cur = st
        steps = 0

        while cur.U:
            A = ready_tasks(cur)
            if not A:
                return self.reward_fail

            t = self.rng.choice(A)

            Vt = self.instance["V_tau"][t]
            if not Vt:
                return self.reward_fail
            v = self.rng.choice(list(Vt))

            cur, _ = self.step(cur, (t, v))

            steps += 1
            if self.n_rollout_limit is not None and steps >= self.n_rollout_limit:
                return self.reward_fail

        obj, delay = compute_obj_delay(self.instance, cur)
        return self._reward(obj, delay)

    def _simulate_sgs_i(self, st: SGSState) -> float:
        cur = st
        steps = 0

        traj = []
        while cur.U:
            A = ready_tasks(cur)
            if not A:
                return self.reward_fail
            A.sort(key=lambda x:(cur.ES[x],x))

            t = A[0]

            if t == 32:
                pass

            Vprime = list()
            E = dict()

            Vt = self.instance["V_tau"][t]
            if not Vt:
                return self.reward_fail
            
            for v1 in Vt:
                R = []
                est = max(cur.ES[t], cur.C[v1] + abs(self.instance["l^1"][t] - cur.L[v1]) * self.instance["hat(t)"])

                if est <= self.instance["ls"][t]:
                    R.append((est, self.instance["ls"][t]))

                    for v2 in Vt:
                        if v2 == v1:
                            continue
                        for t2 in cur.G[v2]:
                            if (t2, t, v2, v1) in self.instance["Theta"]:
                                Rprime = []
                                
                                for (es_t1, ls_t1) in R:
                                    new_ls = min(ls_t1, cur.S[t2] - self.instance['Delta'][(t, t2, v1, v2)] - self.instance['h'][t])
                                    if es_t1 <= new_ls:
                                        Rprime.append((es_t1, new_ls))
                                    
                                    new_es = max(es_t1, cur.S[t2] + self.instance['h'][t2] + self.instance['Delta'][(t2, t, v2, v1)])
                                    if new_es <= ls_t1:
                                        Rprime.append((new_es, ls_t1))
                                
                                R = Rprime
                
                if R:
                    R.sort(key=lambda x:x[0])
                    E[v1] = R[0][0]
                else:

                    for v2 in Vt:
                        if v2 == v1:
                            continue
                        for t2 in cur.G[v2]:
                            if (t2, t, v2, v1) in self.instance["Theta"]:
                                est = max(est, cur.S[t2] + self.instance['h'][t2] + self.instance['Delta'][(t2, t, v2, v1)])
                    E[v1] = est
                
                Vprime.append(v1)
            
            v = min(Vprime, key=lambda x:(E[x],x))

            cur, _ = self.step(cur, (t, v), est=E[v])
            traj.append((t, v, E[v]))

            steps += 1
            if self.n_rollout_limit is not None and steps >= self.n_rollout_limit:
                return self.reward_fail

        obj, delay = compute_obj_delay(self.instance, cur)
        # print(obj, delay)
        # print(traj)
        return self._reward2(obj, delay)

    def _reward(self, obj: float, delay: float) -> float:
        """
        Layered reward with dcap:
          - delay>0 => reward in [-2,-1] (always worse than any delay=0)
          - delay=0 => reward in [0,1] based on obj improvement vs ref
        """
        obj_ref = self.refs["obj_ref"]
        dcap = self.refs["dcap"]
        eps = self.refs["eps"]

        if delay > 0.0:
            penalty = min(1.0, delay / max(dcap, eps))  # [0,1]
            return -1.0 - penalty                        # [-2,-1]
        else:
            imp = (obj_ref - obj) / (abs(obj_ref) + eps)  # improvement ratio
            imp = max(-1.0, min(1.0, imp))                # clip
            return 0.5 + 0.5 * imp                        # [0,1]
    
    def _reward2(self, obj: float, delay: float) -> float:
        """
        Layered reward with dcap:
          - delay>0 => reward in [-2,-1] (always worse than any delay=0)
          - delay=0 => reward in [0,1] based on obj improvement vs ref
        """
        obj_ref = self.refs["obj_ref"]
        dcap = self.refs["dcap"]
        eps = self.refs["eps"]

        # return (obj_ref - obj - delay) / obj_ref
        return - obj - 100 * delay

    def _backprop(self, node: Node, reward: float) -> None:
        cur = node
        while cur is not None:
            cur.N += 1
            cur.W += reward
            cur.Q = cur.W / cur.N
            cur = cur.parent

    def _backprop2(self, node: Node, reward: float) -> None:
        cur = node
        while cur is not None:
            cur.N += 1
            cur.W = max(cur.W, reward)
            cur.Q = cur.W
            cur = cur.parent

    def _choose_final_action(self, root: Node) -> Optional[Action]:
        if not root.children:
            return None
        if self.final_action == "q":
            action = max(root.children.items(), key=lambda kv: kv[1].Q)[0]
            return action, root.children[action].expended_est
        action = max(root.children.items(), key=lambda kv: kv[1].N)[0]
        return action, root.children[action].expended_est
