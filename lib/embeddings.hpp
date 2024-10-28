/* Meant to be a helper file to generate embeddings and the ground truth */
#pragma once

#include <cassert>
#include <cstdint>
#include <memory>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

#include <faiss/IndexFlat.h>

/// @brief Embedding struct container vector embedding data as well as associated metadata
struct Embedding {
	const std::unique_ptr<float[]> data;
	const size_t nb;
	const size_t dim;
};

/// @brief generate vector embeddings sampled from a seeded uniform distribution in contiguous memory
/// @param nb number of vectors
/// @param dim dimension of each vector
/// @return flattened pointer of the vectors
Embedding generate_test_vectors(size_t nb, size_t dim, uint_fast32_t seed = 47) {
	std::mt19937 rng;
	rng.seed(47);

	std::uniform_real_distribution<> uniform_distribution;

	std::unique_ptr<float[]> ret = std::make_unique<float[]>(nb * dim);
	for(size_t i = 0; i < dim * nb; i++) {
		ret[i] = uniform_distribution(rng);
	}

	return { std::move(ret), nb, dim };
}

/// @brief calculate the ground truth using the faiss library which should be reasonably fast
/// @param embedding vector embeddings
/// @param query query vector that has the same dimension as that of the vector embedding
/// @param top_k top k nearest neighbors
/// @return array of distances and indicies
std::pair<std::unique_ptr<float[]>, std::unique_ptr<faiss::idx_t[]>>
get_top_k_euclidean(const Embedding& embedding, const float* query, size_t top_k) {
	assert(top_k >= embedding.nb);

	// use faiss to get ground truth
	faiss::IndexFlatL2 index(embedding.dim);
	index.add(embedding.nb, embedding.data.get());

	std::unique_ptr<float[]> distances = std::make_unique<float[]>(top_k);
	std::unique_ptr<faiss::idx_t[]> indices = std::make_unique<faiss::idx_t[]>(top_k);

	index.search(1, query, top_k, distances.get(), indices.get());
	return std::make_pair(std::move(distances), std::move(indices));
}

double calculate_recall(const Embedding& embedding,
						const std::vector<size_t> results,
						const float* query) {

	const size_t top_k = results.size();
	auto [_, indicies] = get_top_k_euclidean(embedding, query, top_k);

	std::vector<bool> contains(embedding.nb, false);
	for(size_t i = 0; i < top_k; i++) {
		contains[indicies[i]] = true;
	}

	size_t num_hits = 0;
	for(const size_t& item : results) {
		assert(item < embedding.nb);
		if(contains[item]) {
			num_hits++;
		}
	}

	return static_cast<double>(num_hits) / top_k;
}
