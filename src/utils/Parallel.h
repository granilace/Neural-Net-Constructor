#pragma once
#include <thread>
#include <functional>
const int N_THREADS = 4;

void parallelize(int bs, std::function<void(int, int)> f);