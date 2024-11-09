#include "lib/argparser.hpp"
#include "lib/embeddings.hpp"
#include "lib/utils.hpp"

#include <cassert>
#include <chrono>
#include <filesystem>
#include <format>
#include <hnswlib/hnswlib.h>
#include <random>
#include <vector>

namespace chrono = std::chrono;
namespace fs = std::filesystem;

// CONFIGURE HNSW
int EF[] = { 100, 150, 200, 250, 300 };

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
inline constexpr size_t NUM_SINGLE_QUERIES = 5;

// the K to run the single query on
inline constexpr size_t SINGLE_QUERY_K = 100;

inline constexpr int num_threads = 32;
int main(int argc, char** argv) {

	// parse arguments because I'm cool
	argparse::ArgumentParser program("bench_st_sq");

	program.add_argument("gist_dir").help("path to base gist directory");
	program.add_argument("res_path").help("path to directory to write result");
	program.add_argument("index_path").help("path to hnsw index file");

	try {
		program.parse_args(argc, argv);
	} catch(const std::exception& err) {
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		return 1;
	}

	const fs::path gist_dir{ program.get<std::string>("gist_dir") };
	const fs::path res_path{ program.get<std::string>("res_path") };
	const fs::path index_path{ program.get<std::string>("index_path") };

	const fs::path gist_query = gist_dir / "gist_query.fvecs";
	const fs::path gist_groundtruth = gist_dir / "gist_groundtruth.ivecs";

	std::cout << "Configurations: " << std::endl;
	std::cout << std::format("\tNUM_QUERIES = {}", NUM_SINGLE_QUERIES) << std::endl;
	std::cout << std::format("\tITERS_PER_QUERY = {}", RUNS_FOR_SINGLE_QUERY) << std::endl;
	std::cout << std::format("\tTOP_K= {}", SINGLE_QUERY_K) << std::endl;

	const auto GIST_Q = load_gist_960<float>(gist_query);
	std::cout << std::format("gist query with NB = {} and DIM = {}", GIST_Q.nb, GIST_Q.dim)
			  << std::endl;

	const auto GIST_GT = load_gist_960<int>(gist_groundtruth);
	std::cout << std::format("gist gt with NB = {} and DIM = {}", GIST_GT.nb, GIST_GT.dim)
			  << std::endl;

	assert(NUM_SINGLE_QUERIES <= GIST_Q.nb);

	std::cout << std::format("loading from file: {}", index_path.string()) << std::endl;
	hnswlib::L2Space space(960);
	hnswlib::HierarchicalNSW<float> alg_hnsw = hnswlib::HierarchicalNSW<float>(&space, index_path);

	// Test 1: performance querying a single query multiple times

	// rows correspond to query id and columns correspond to the latency for that run
	for(int ef : EF) {
		uint64_t single_query_latency[NUM_SINGLE_QUERIES][RUNS_FOR_SINGLE_QUERY];
		double single_query_recall[NUM_SINGLE_QUERIES];

		for(size_t test_id = 0; test_id < NUM_SINGLE_QUERIES; test_id++) {
			std::cout << std::format("run id: {} ef: {}", test_id, ef) << std::endl;
			std::priority_queue<std::pair<float, hnswlib::labeltype>> output;

			// std::cout << "Q: " << GIST_Q.dim << " " << GIST_Q.nb << std::endl;
			const float* vector_addr =
				static_cast<float*>(GIST_Q.data.get() + GIST_Q.dim * test_id);
			for(size_t run_id = 0; run_id < RUNS_FOR_SINGLE_QUERY; run_id++) {
				auto start = chrono::high_resolution_clock::now();
				auto o = alg_hnsw.searchKnn(vector_addr, SINGLE_QUERY_K);
				auto end = chrono::high_resolution_clock::now();
				auto duration_elapsed = chrono::duration_cast<chrono::microseconds>(end - start);
				uint64_t duration_elapsed_us = duration_elapsed.count();
				single_query_latency[test_id][run_id] = duration_elapsed_us;
				output = std::move(o);
			}

			single_query_recall[test_id] = calculate_recall(test_id, GIST_GT, output);
			std::cout << "\trecall: " << (single_query_recall[test_id] * 100) << "%" << std::endl;
		}

		fs::path csv_filename =
			res_path / fs::path(std::format(
						   "1-ST-CPU_dim_960_nb_1000000_{}_searchef_{}_same_vector_latencies.csv",
						   index_path.filename().string(),
						   ef));

		std::cout << "writing to file: " << csv_filename.string() << std::endl;
		std::ofstream fout(csv_filename);
		if(fout.is_open()) {
			fout << "id";
			for(int iter = 0; iter < RUNS_FOR_SINGLE_QUERY; iter++)
				fout << ", iter" << (iter + 1) << " (us)";

			fout << ", recall";
			fout << '\n';
			for(size_t test_id = 0; test_id < NUM_SINGLE_QUERIES; test_id++) {
				fout << test_id;

				for(int iter = 0; iter < RUNS_FOR_SINGLE_QUERY; iter++) {
					fout << ", " << single_query_latency[test_id][iter];
				}
				fout << ", " << single_query_recall[test_id];
				fout << '\n';
			}
			fout.close();
		} else {
			std::cerr << "cannot open file: " << csv_filename << std::endl;
		}
	}
	return 0;
}