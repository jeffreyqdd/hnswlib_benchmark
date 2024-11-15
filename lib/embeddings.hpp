/* Meant to be a helper file to generate embeddings and the ground truth */
#pragma once

#include <cassert>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <hnswlib/hnswlib.h>
#include <memory>
#include <queue>
#include <unordered_set>

template <typename T>
struct Embedding {
	const std::unique_ptr<T[]> data;
	const int dim;
	const int nb;
};

/// @brief read gist dataset
/// @param gist_src path to gist fvecs file
/// @param dim dimension of dataset is stored in dim
/// @param nb number of vectors is stored in nb
/// @return
template <typename T>
Embedding<T> load_gist_960(const std::filesystem::path& gist_src) {
	std::ifstream fin(gist_src, std::ios::binary);

	if(!fin) {
		throw std::runtime_error(std::format("could not open filename {}", gist_src.string()));
	}
	int dim;
	int nb;
	fin.read(reinterpret_cast<char*>(&dim), sizeof(T));

	if(!fin) {
		throw std::runtime_error("could not read dimensions");
	}

	const size_t file_size = std::filesystem::file_size(gist_src);
	nb = file_size / (dim * sizeof(T) + sizeof(int));

	std::unique_ptr<T[]> data = std::make_unique<T[]>(nb * dim);
	fin.seekg(0, std::ios::beg);
	for(size_t i = 0; i < static_cast<size_t>(nb); i++) {
		// read dim
		int tmp_dim;
		fin.read(reinterpret_cast<char*>(&tmp_dim), sizeof(int));
		assert(tmp_dim == dim);

		//read vector
		fin.read(reinterpret_cast<char*>(data.get() + i * dim), tmp_dim * sizeof(T));
	}

	return { std::move(data), dim, nb };
}

double calculate_recall(const int query_id,
						const Embedding<int>& ground_truth,
						std::priority_queue<std::pair<float, hnswlib::labeltype>>& results) {

	const int* truth_ptr = ground_truth.data.get();

	std::unordered_set<int> contains;
	/// 100, 1000

	// std::cout << ground_truth.dim << " " << ground_truth.nb << std::endl;
	for(int i = 0; i < ground_truth.dim; i++) {
		const int* vector_id = truth_ptr + (query_id * ground_truth.dim + i);
		contains.insert(*vector_id);
	}

	int num_hits = 0;
	assert(contains.size() == 100);
	// std::cout << contains.size() << std::endl;
	while(!results.empty()) {
		auto [dist, label] = results.top();
		results.pop();

		if(contains.contains(label)) {
			num_hits++;
		}
	}

	return static_cast<double>(num_hits) / static_cast<double>(ground_truth.dim);
}