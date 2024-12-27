import numpy as np
import random

class GA:
    def __init__(self, pop_size, gens, mut_rate, cross_rate, tour_size, elit, seed, obj='total_time'):
        self.pop_size = pop_size  # 種群大小
        self.gens = gens          # 迭代次數
        self.mut_rate = mut_rate  # 突變率
        self.cross_rate = cross_rate  # 交叉率
        self.tour_size = tour_size    # 規模
        self.elit = elit          # 是否使用精英策略
        self.obj = obj            # 目標函數：'total_time' 或 'makespan'

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def init_pop(self, M, N):
        if N < M:
            raise ValueError("任務數必須大於或等於人員數")
        pop = []
        for _ in range(self.pop_size):
            indiv = [None] * N
            tasks = list(range(N))
            random.shuffle(tasks)
            for stu_id in range(M):
                task = tasks.pop()
                indiv[task] = stu_id
            for task in tasks:
                stu_id = random.randint(0, M - 1)
                indiv[task] = stu_id
            pop.append(indiv)
        return pop

    def fitness(self, indiv, times):
        stu_ids = np.array(indiv)  # 每個任務對應的人員編號
        task_idxs = np.arange(len(indiv))  # 任務索引
        times_per_task = times[stu_ids, task_idxs]  # 每個任務的完成時間
        M = times.shape[0]
        stu_total_times = np.bincount(stu_ids, weights=times_per_task, minlength=M)

        if self.obj == 'makespan':
            total_time = stu_total_times.max()
        elif self.obj == 'total_time':
            total_time = stu_total_times.sum()
        else:
            raise ValueError(f"未知目標: {self.obj}")

        return -total_time  # 取負數，方便最大化

    def selection(self, pop, fits):
        idxs = random.sample(range(len(pop)), self.tour_size)
        indivs = [pop[i] for i in idxs]
        indiv_fits = [fits[i] for i in idxs]
        best_idx = np.argmax(indiv_fits)
        return indivs[best_idx]

    def ensure_valid(self, indiv, M):
        assigned = set(indiv)
        missing = set(range(M)) - assigned
        counts = {s: indiv.count(s) for s in assigned}
        if missing:
            for stu in missing:
                donors = [s for s, c in counts.items() if c > 1]
                if not donors:
                    donors = [s for s, c in counts.items() if c > 0]
                donor = random.choice(donors)
                if counts[donor] == 1 and len(assigned) == M - len(missing):
                    continue
                tasks_idxs = [i for i, s in enumerate(indiv) if s == donor]
                task_to_reassign = random.choice(tasks_idxs)
                indiv[task_to_reassign] = stu
                counts[donor] -= 1
                counts[stu] = counts.get(stu, 0) + 1
                assigned.add(stu)
                if counts[donor] == 0:
                    assigned.remove(donor)
        return indiv

    def crossover(self, p1, p2, M):
        if random.random() < self.cross_rate:
            point = random.randint(1, len(p1) - 2)
            o1 = p1[:point] + p2[point:]
            o2 = p2[:point] + p1[point:]
        else:
            o1 = p1[:]
            o2 = p2[:]
        o1 = self.ensure_valid(o1, M)
        o2 = self.ensure_valid(o2, M)
        return o1, o2

    def mutate(self, indiv, M):
        for i in range(len(indiv)):
            if random.random() < self.mut_rate:
                indiv[i] = random.randint(0, M - 1)
        indiv = self.ensure_valid(indiv, M)
        return indiv

    def __call__(self, M, N, times):
        pop = self.init_pop(M, N)
        fits = [self.fitness(indiv, times) for indiv in pop]

        for _ in range(self.gens):
            new_pop = []
            if self.elit:
                elite_idx = np.argmax(fits)
                elite = pop[elite_idx]
                new_pop.append(elite)

            while len(new_pop) < self.pop_size:
                p1 = self.selection(pop, fits)
                p2 = self.selection(pop, fits)
                o1, o2 = self.crossover(p1, p2, M)
                o1 = self.mutate(o1, M)
                o2 = self.mutate(o2, M)
                new_pop.extend([o1, o2])

            pop = new_pop[:self.pop_size]
            fits = [self.fitness(indiv, times) for indiv in pop]

        best_idx = np.argmax(fits)
        best_indiv = pop[best_idx]
        total_time = -fits[best_idx]
        return best_indiv, int(round(total_time))


if __name__ == "__main__":
    def write_output(problem_num, total_time, filename="answer.txt"):
        print(f"Total time = {total_time}")
        with open(filename, 'a') as f:
            f.write(f"Total time = {total_time}\n")

    # 定義問題
    M1, N1 = 2, 3
    cost1 = [
        [3, 2, 4],
        [4, 3, 2]
    ]

    M2, N2 = 4, 4
    cost2 = [
        [5, 4, 6, 3],
        [6, 5, 4, 2],
        [7, 6, 5, 4],
        [4, 3, 2, 5]
    ]

    M3, N3 = 8, 9
    cost3 = [
        [90, 100, 60, 5, 50, 1, 100, 80, 70],
        [100, 5, 90, 100, 50, 70, 60, 90, 100],
        [50, 1, 100, 70, 90, 60, 80, 100, 4],
        [60, 100, 1, 80, 70, 90, 100, 50, 100],
        [70, 90, 50, 100, 100, 4, 1, 60, 80],
        [100, 60, 100, 90, 80, 5, 70, 100, 50],
        [100, 4, 80, 100, 90, 70, 50, 1, 60],
        [1, 90, 100, 50, 60, 80, 100, 70, 5]
    ]

    M4, N4 = 3, 3
    cost4 = [
        [2, 5, 6],
        [4, 3, 5],
        [5, 6, 2]
    ]

    M5, N5 = 4, 4
    cost5 = [
        [5, 4, 6, 1],
        [1, 9, 6, 2],
        [9, 6, 5, 3],
        [4, 2, 2, 5]
    ]

    M6, N6 = 4, 4
    cost6 = [
        [5, 4, 6, 7],
        [8, 3, 4, 6],
        [6, 7, 3, 8],
        [7, 8, 9, 2]
    ]

    M7, N7 = 4, 4
    cost7 = [
        [25 * 0.16, 24 * 0.292, 23 * 0.348, 25 * 0.36],
        [25 * 0.24, 24 * 0.125, 23 * 0.261, 25 * 0.28],
        [25 * 0.32, 24 * 0.250, 23 * 0.087, 25 * 0.24],
        [25 * 0.28, 24 * 0.333, 23 * 0.304, 25 * 0.12]
    ]

    M8, N8 = 5, 5
    cost8 = [
        [8, 8, 24, 24, 24],
        [6, 18, 6, 18, 18],
        [30, 10, 30, 10, 30],
        [21, 21, 21, 7, 7],
        [27, 27, 9, 27, 9]
    ]

    M9, N9 = 5, 5
    cost9 = [
        [10, 10, None, None, None],
        [12, None, None, None, 12],
        [None, 15, 15, None, None],
        [11, None, 11, None, None],
        [None, 14, None, 14, None]
    ]

    cost9 = np.array(cost9, dtype=float)
    cost9[np.isnan(cost9)] = np.inf

    M10, N10 = 9, 10
    cost10 = [
        [1, 90, 100, 50, 70, 20, 100, 60, 80, 90],
        [100, 10, 1, 100, 60, 80, 70, 100, 50, 90],
        [90, 50, 70, 1, 100, 100, 60, 90, 80, 100],
        [70, 100, 90, 5, 10, 60, 100, 80, 90, 50],
        [50, 100, 100, 90, 20, 4, 80, 70, 60, 100],
        [100, 5, 80, 70, 90, 100, 4, 50, 1, 60],
        [90, 60, 50, 4, 100, 90, 100, 5, 10, 80],
        [100, 70, 90, 100, 4, 60, 1, 90, 100, 5],
        [80, 100, 5, 60, 50, 90, 70, 100, 4, 1]
    ]



    problems = [
        (M1, N1, np.array(cost1)),
        (M2, N2, np.array(cost2)),
        (M3, N3, np.array(cost3)),
        (M4, N4, np.array(cost4)),
        (M5, N5, np.array(cost5)),
        (M6, N6, np.array(cost6)),
        (M7, N7, np.array(cost7)),
        (M8, N8, np.array(cost8)),
        (M9, N9, np.array(cost9)),
        (M10, N10, np.array(cost10))
    ]

    # 設定目標為 'total_time'
    objectives = ['total_time'] * len(problems)

    # 清空輸出文件
    open('answer.txt', 'w').close()

    # 解決每個問題並寫入文件
    for i, ((M, N, times), obj) in enumerate(zip(problems, objectives), 1):
        ga = GA(
            pop_size=200,
            gens=500,
            mut_rate=0.1,
            cross_rate=0.9,
            tour_size=10,
            elit=True,
            seed=42,
            obj=obj
        )

        best_alloc, total_time = ga(M=M, N=N, times=times)
        write_output(i, total_time)

    print("結果已寫入 answer.txt")
