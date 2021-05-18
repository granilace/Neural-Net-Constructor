#include "Parallel.h"

void parallelize(int bs, std::function<void(int, int)> f) {
    int n_threads = N_THREADS;
    if (bs < n_threads)
        n_threads = bs;
    std::vector<std::thread> threads;
    for (int tid = 0; tid < n_threads; tid++) {
        threads.emplace_back(f, tid, n_threads);
    }
    for (auto & thread : threads) {
        thread.join();
    }
}