#include "lib/embeddings.hpp"
#include "lib/stopwatch.hpp"
#include "lib/utils.hpp"

#include <cassert>
#include <chrono>
#include <filesystem>
#include <hnswlib/hnswlib.h>
#include <random>
#include <vector>

namespace chrono = std::chrono;
namespace fs = std::filesystem;

// CONFIGURE HNSW
inline constexpr int M = 32;
inline constexpr int dim = 960;
inline constexpr int nb = 1000;
inline constexpr int ef_construction = 128;
inline constexpr int ef = 64;

// We want to benchmark a few items
//
// 1. performance querying a single query multiple times
// 2. performance querying different queries multiple times
// 3. recall of querying different queries of top k = 100, and possibly other top k's
//
// Note: batching is not really applicable to HNSWs

// CONFIGURE BENCHMARK VALUES
// number of runs for a single vector
inline constexpr size_t RUNS_FOR_SINGLE_QUERY = 1000;

// number of different vectors to try single queries on
inline constexpr size_t NUM_SINGLE_QUERIES = 20;

// the K to run the single query on
inline constexpr size_t SINGLE_QUERY_K = 100;

// number of runs for each top K
inline constexpr size_t RUNS_FOR_TOP_K = 1000;

// number of K's to try
inline const std::vector<size_t> TOP_KS{ 1, 2, 5, 10, 20, 50, 100 };

inline constexpr int num_threads = 32;
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

	std::cout << "generating test vectors graph" << std::endl;
	const Embedding test_vectors = generate_test_vectors(nb, dim);

	hnswlib::L2Space space(dim);
	hnswlib::HierarchicalNSW<float>* alg_hnsw =
		new hnswlib::HierarchicalNSW<float>(&space, nb, M, ef_construction);
	alg_hnsw->setEf(ef);

	// Add data to the index
	std::cout << "populating graph" << std::endl;
	ParallelFor(0, nb, num_threads, [&test_vectors, &alg_hnsw](size_t row, size_t thread_id) {
		if(row % 100 == 0)
			std::cout << '\r' << row << "             ";
		alg_hnsw->addPoint((void*)(test_vectors.data.get() + dim * row), row);
	});
	std::cout << std::endl;

	// Test 1: performance querying a single query multiple times

	// rows correspond to query id and columns correspond to the latency for that run
	assert(NUM_SINGLE_QUERIES < nb);
	uint64_t single_query_latency[NUM_SINGLE_QUERIES][RUNS_FOR_SINGLE_QUERY];

	for(size_t test_id = 0; test_id < NUM_SINGLE_QUERIES; test_id++) {
		std::cout << "running single query test for id = " << test_id << std::endl;
		for(size_t run_id = 0; run_id < RUNS_FOR_SINGLE_QUERY; run_id++) {
			auto start = chrono::high_resolution_clock::now();
			alg_hnsw->searchKnn(static_cast<void*>(test_vectors.data.get() + dim * test_id),
								SINGLE_QUERY_K);
			auto end = chrono::high_resolution_clock::now();
			auto duration_elapsed = chrono::duration_cast<chrono::microseconds>(end - start);
			uint64_t duration_elapsed_us = duration_elapsed.count();
			single_query_latency[test_id][run_id] = duration_elapsed_us;
		}
	}

	fs::path base_dir = fs::path(result_path);
	fs::path csv_filename =
		base_dir / fs::path("1-ST-CPU_dim_" + std::to_string(dim) + "_nb_" + std::to_string(nb) +
							"_M_" + std::to_string(M) + "_ef_" + std::to_string(ef_construction) +
							"_K_" + std::to_string(SINGLE_QUERY_K) + "_same_vector_latencies.csv");

	std::ofstream fout(csv_filename);
	if(fout.is_open()) {
		fout << "id";
		for(int iter = 0; iter < RUNS_FOR_SINGLE_QUERY; iter++)
			fout << ", iter" << (iter + 1) << " (us)";

		fout << '\n';
		for(size_t test_id = 0; test_id < NUM_SINGLE_QUERIES; test_id++) {
			fout << test_id;

			for(int iter = 0; iter < RUNS_FOR_SINGLE_QUERY; iter++) {
				fout << ", " << single_query_latency[test_id][iter];
			}

			fout << '\n';
		}
		fout.close();
	} else {
		std::cerr << "cannot open file: " << csv_filename << std::endl;
	}
	return 0;
}