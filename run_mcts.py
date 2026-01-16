from __future__ import annotations

import csv
import os
import time

from utils.load_instance import load_instance
from sgs.sgs_ops import build_initial_state, compute_obj_delay
from mcts.refs import compute_refs
from mcts.core import MCTS

from tqdm import tqdm



def inst_path(inst_dir: str, n_cranes: int, n_jobs: int, idx: int) -> str:
    return os.path.join(inst_dir, f"{n_cranes}_{n_jobs}_{idx}.json")


def append_row(csv_path: str, row: dict) -> None:
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)


def run_one(inst_file: str, seed: int, iters_per_move: int, c_uct: float, final_action: str):
    t0 = time.time()
    inst = load_instance(inst_file, apply_rules=True)

    # refs (obj_ref, dcap, ...)
    refs = compute_refs(inst, seed=seed)

    # MCTS constructive
    st = build_initial_state(inst)
    mcts = MCTS(inst, refs, seed=seed, c_uct=c_uct, final_action=final_action)

    ok = True
    for _ in tqdm(range(len(inst["T"]))):
        action, est = mcts.search(st, n_iter=iters_per_move)
        if action is None:
            ok = False
            break
        st, _ = mcts.step(st, action, est=est)
        # print(action)
        if action == (14,4):
            pass
    assert st.U == set()

    if ok and not st.U:
        mcts_obj, mcts_delay = compute_obj_delay(inst, st)
    else:
        mcts_obj, mcts_delay = float("inf"), float("inf")

    wall = time.time() - t0
    return ok, refs['obj_ref'], 0, mcts_obj, mcts_delay, wall



# =========================
# ✅ EXPERIMENT CONFIG (여기만 바꾸면 됨)
# =========================
CFG = {
    "inst_dir": "./instances",

    # 실험 조합
    "n_cranes_list": [2],
    "n_jobs_list": [20],
    "idx_list": range(30),

    # MCTS params
    "iters_per_move": 10,
    "c_uct": 0,
    "final_action": "q",  # "visits" or "q"

    # seed policy
    "seed": 7,
    "seed_by_idx": True,  # seed + idx

    # output
    "out_csv": "results.csv",
}
# =========================

def main():
    inst_dir = CFG["inst_dir"]
    out_csv = CFG["out_csv"]

    for ncr in CFG["n_cranes_list"]:
        for nj in CFG["n_jobs_list"]:
            for idx in CFG["idx_list"]:
                f = inst_path(inst_dir, ncr, nj, idx)
                if not os.path.exists(f):
                    print(f"[MISSING] {f}")
                    continue

                seed = CFG["seed"] + idx if CFG["seed_by_idx"] else CFG["seed"]
                ok, base_obj, base_delay, mcts_obj, mcts_delay, wall = run_one(
                    inst_file=f,
                    seed=seed,
                    iters_per_move=CFG["iters_per_move"],
                    c_uct=CFG["c_uct"],
                    final_action=CFG["final_action"],
                )

                imp = (base_obj - mcts_obj) / abs(base_obj) if (ok and base_obj not in [0.0, float("inf")]) else 0.0

                row = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "file": os.path.basename(f),
                    "n_cranes": ncr,
                    "n_jobs": nj,
                    "idx": idx,
                    "seed": seed,
                    "iters_per_move": CFG["iters_per_move"],
                    "c_uct": CFG["c_uct"],
                    "final_action": CFG["final_action"],
                    "ok": ok,
                    "base_obj": base_obj,
                    "base_delay": base_delay,
                    "mcts_obj": mcts_obj,
                    "mcts_delay": mcts_delay,
                    "imp_ratio": imp,
                    "wall_sec": wall,
                }
                append_row(out_csv, row)

                print(
                    f"[{row['file']}] ok={ok} base={base_obj:.6f} mcts={mcts_obj:.6f} "
                    f"imp={imp:.6f} delay={mcts_delay:.6f} wall={wall:.2f}s"
                )


if __name__ == "__main__":
    main()
