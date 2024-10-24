#include "lib/stopwatch.hpp"
#include "lib/utils.hpp"
#include <chrono>
#include <filesystem>
#include <hnswlib/hnswlib.h>
#include <random>
#include <vector>

namespace chrono = std::chrono;
namespace fs = std::filesystem;

// CONFIGURE HNSW
inline constexpr int M = 48;
inline constexpr int dim = 960;
inline constexpr int nb = 1000000;
inline constexpr int ef_construction = 512;
inline constexpr int ef = 256;

// CONFIGURE BENCHMARK VALUES
inline constexpr size_t N = 10;
inline constexpr size_t K = 100;
inline constexpr size_t num_searches = 1000;
inline constexpr size_t nk[N] = { 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000 };

inline constexpr int num_threads = 30;
int main(int argc, char** argv) {

	int opt;
	std::string result_path;
	while((opt = getopt(argc, argv, "p:")) != -1) {
		switch(opt) {
		case 'p':
			result_path = optarg;
		default:
			break;
		}
	}

	hnswlib::L2Space space(dim);
	hnswlib::HierarchicalNSW<float>* alg_hnsw =
		new hnswlib::HierarchicalNSW<float>(&space, nb, M, ef_construction);
	alg_hnsw->setEf(ef);

	// generate randomized data
	std::mt19937 rng;
	rng.seed(47);
	std::uniform_real_distribution<> distrib_real;
	float* data = new float[dim * nb];
	for(int i = 0; i < dim * nb; i++) {
		data[i] = distrib_real(rng);
	}

	// Add data to the index
	ParallelFor(0, nb, num_threads, [&data, &alg_hnsw](size_t row, size_t thread_id) {
		alg_hnsw->addPoint((void*)(data + dim * row), row);
	});

	// Latency Storage: where rows are nq configurations and columns are iterations
	uint64_t latency[N][num_searches];
	double recall[N];

	assert(dim * num_searches < nb);
	for(size_t test_id = 0; test_id < N; test_id++) {
		std::cout << "running test: " << test_id << ": " << nk[test_id] << std::endl;
		size_t num_correct = 0;
		uint64_t sum = 0;

		for(size_t it = 0; it < num_searches; it++) {
			// query the batch
			auto start = chrono::high_resolution_clock::now();

			auto pq = alg_hnsw->searchKnn(static_cast<void*>(data + dim * it), nk[test_id]);

			auto end = chrono::high_resolution_clock::now();
			auto duration_elapsed = chrono::duration_cast<chrono::microseconds>(end - start);
			uint64_t duration_elapsed_us = duration_elapsed.count();

			latency[test_id][it] = duration_elapsed_us;

			while(!pq.empty()) {
				auto [dist, vector_id] = pq.top();
				pq.pop();
				if(vector_id == it) {
					num_correct += 1;
					break;
				}
			}
			sum += duration_elapsed_us;
		}
		recall[test_id] = static_cast<double>(num_correct) / static_cast<double>(num_searches);
		std::cout << "\t correct: " << num_correct << " total: " << num_searches << std::endl;
		std::cout << "\t estimated avg time (us): " << (sum / num_searches) << std::endl;
		std::cout << "\t recall (%): " << (recall[test_id] * 100) << std::endl;
	}

	delete alg_hnsw;
	delete[] data;
	fs::path base_dir = fs::path(result_path);
	fs::path csv_filename =
		base_dir /
		fs::path("1-ST-CPU_dim_" + std::to_string(dim) + "_nb_" + std::to_string(nb) + "_M_" +
				 std::to_string(M) + "_ef_" + std::to_string(ef_construction) + "_latencies.csv");

	std::cout << "creating csv: " << csv_filename << std::endl;

	std::ofstream fout(csv_filename);
	if(fout.is_open()) {
		fout << "k";
		for(int iter = 0; iter < num_searches; iter++)
			fout << ", iter" << (iter + 1) << " (us)";
		fout << ", recall (%)\n";

		for(size_t nq_idx = 0; nq_idx < N; nq_idx++) {
			fout << nk[nq_idx];
			for(int iter = 0; iter < num_searches; iter++) {
				fout << ", " << latency[nq_idx][iter];
			}
			fout << ", " << recall[nq_idx] << '\n';
		}
		fout.close();
	} else {
		std::cerr << "cannot open file: " << csv_filename << std::endl;
	}

	return 0;
}