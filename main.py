import time
from tqdm import tqdm
from utils import load_instance
from init_heuristics import sgs_a




def main(n_cranes, n_jobs, instance_idx, n_sampling=1):
    # 인스턴스 경로 생성
    inst_path = f"./instances/{n_cranes}_{n_jobs}_{instance_idx}.json"
    # 로딩
    instance = load_instance(inst_path)

    results = []
    for _ in tqdm(range(n_sampling)):
    # for _ in range(n_sampling):
        # 휴리스틱 실행
        start = time.time()
        schedule, assignment, obj, delay = sgs_a(instance)
        runtime = time.time() - start

        # 결과 출력
        # print(f"[INFO] Instance: C={n_cranes}, J={n_jobs}, idx={instance_idx}")
        # print(f"obj={obj}, delay={delay}, runtime={runtime:.4f} sec")
        # print()

        results.append((obj, delay))
    results.sort(key=lambda x: (x[1], x[0]))
    print(instance_idx, results[0])


if __name__ == "__main__":
    # for n_cranes in [2, 3]:
    #     if n_cranes == 2 :
    #         job_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    #     else :
    #         job_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    #     for n_jobs in job_list :
    #         for instance_idx in range(30):
    #             main(
    #                 n_cranes=n_cranes,
    #                 n_jobs=n_jobs,
    #                 instance_idx=instance_idx,
    #             )
    
    for idx in range(30):
        main(
            n_cranes=2,
            n_jobs=100,
            instance_idx=idx,
            n_sampling=1000
        )
