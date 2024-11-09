
#include "lib/argparser.hpp"
#include "lib/embeddings.hpp"
#include "lib/utils.hpp"

#include <cmath>
#include <execution>
#include <filesystem>
#include <format>
#include <hnswlib/hnswlib.h>
#include <numeric>

namespace chrono = std::chrono;
namespace fs = std::filesystem;

constexpr int NUM_THREADS = 20;

template <typename T>
struct std::formatter<std::vector<T>> {
	constexpr auto parse(std::format_parse_context& ctx) {
		return ctx.begin();
	}

	auto format(const std::vector<T>& vec, std::format_context& ctx) const {
		auto out = ctx.out();
		*out++ = '{';
		for(auto it = vec.begin(); it != vec.end(); ++it) {
			if(it != vec.begin()) {
				*out++ = ',';
				*out++ = ' ';
			}
			out = std::format_to(out, "{}", *it);
		}
		*out++ = '}';
		return out;
	}
};

void fast_normalize(const float* src, float* dest, size_t dim) {
	float norm = std::inner_product(src, src + dim, src, 0.0f);
	norm = 1.0f / (std::sqrt(norm) + std::numeric_limits<float>::epsilon());

	std::transform(std::execution::seq, src, src + dim, dest, [norm](float x) { return x * norm; });
}

void build_hnsw(hnswlib::HierarchicalNSW<float>& hnsw,
				const Embedding<float>& embedding,
				bool normalize) {

	const float* data = embedding.data.get();

	float normalized_point[NUM_THREADS][960];

	ParallelFor(0, embedding.nb, NUM_THREADS, [&](size_t row, size_t id) {
		const float* point = data + embedding.dim * row;

		if(normalize) {
			fast_normalize(point, normalized_point[id], embedding.dim);
			hnsw.addPoint(point, row);
		} else {
			hnsw.addPoint(point, row);
		}
	});
}

int main(int argc, char** argv) {
	argparse::ArgumentParser program("bench_st_sq");

	program.add_argument("gist_dir").help("path to base gist directory");
	program.add_argument("index_path").help("path to directory to hnswlib indexes");
	program.add_argument("-m")
		.help("list of space separated m's")
		.scan<'i', int>()
		.nargs(argparse::nargs_pattern::at_least_one);
	program.add_argument("-e")
		.help("list of space separated e's for construction")
		.scan<'i', int>()
		.nargs(argparse::nargs_pattern::at_least_one);

	program.add_argument("--use-euclidean")
		.help("build using l2 distance")
		.default_value(false)
		.implicit_value(true);

	program.add_argument("--use-cosine")
		.help("build using cosine distance (normalizes the vectors)")
		.default_value(false)
		.implicit_value(true);

	try {
		program.parse_args(argc, argv);
	} catch(const std::exception& err) {
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		return 1;
	}

	const fs::path gist_dir{ program.get<std::string>("gist_dir") };
	const fs::path gist_base{ gist_dir / "gist_base.fvecs" };
	const fs::path index_path{ program.get<std::string>("index_path") };

	const std::vector<int> hyperparams_m = program.get<std::vector<int>>("-m");
	const std::vector<int> hyperparams_e = program.get<std::vector<int>>("-e");
	const bool use_euclidean = program.get<bool>("--use-euclidean");
	const bool use_cosine = program.get<bool>("--use-cosine");

	std::cout << std::format("HNSW Building Settings") << std::endl;
	std::cout << std::format("\t gist path: '{}'", gist_dir.string()) << std::endl;
	std::cout << std::format("\t gist base: '{}'", gist_base.string()) << std::endl;
	std::cout << std::format("\t index save path: '{}'", index_path.string()) << std::endl;
	std::cout << std::format("\t M's to build: {}", hyperparams_m) << std::endl;
	std::cout << std::format("\t Ef construction's to build: {}", hyperparams_e) << std::endl;
	std::cout << std::format("\t build l2: {}", use_euclidean) << std::endl;
	std::cout << std::format("\t build cosine: {}", use_cosine) << std::endl;

	assert(fs::exists(gist_dir) && fs::is_directory(gist_dir));
	assert(fs::exists(gist_base) && fs::is_regular_file(gist_base));
	assert(fs::exists(index_path) && fs::is_directory(index_path));

	auto gist_vectors = load_gist_960<float>(gist_base);

	for(const int m : hyperparams_m) {
		for(const int ef_construction : hyperparams_e) {
			if(use_euclidean) {
				fs::path save_file =
					index_path / std::format("hnsw_m_{}_ef_{}_l2.bin", m, ef_construction);
				if(fs::exists(save_file)) {
					std::cout << std::format("skipping index: {}", save_file.string()) << std::endl;
				} else {
					std::cout << std::format("generating index: {}", save_file.string())
							  << std::endl;

					hnswlib::L2Space l2_space(gist_vectors.dim);
					hnswlib::HierarchicalNSW<float> alg_hnsw = hnswlib::HierarchicalNSW<float>(
						&l2_space, gist_vectors.nb, m, ef_construction);
					build_hnsw(alg_hnsw, gist_vectors, false);

					alg_hnsw.saveIndex(save_file.string());
					std::cout << std::endl;
				}
			}

			if(use_cosine) {
				fs::path save_file =
					index_path / std::format("hnsw_m_{}_ef_{}_cos.bin", m, ef_construction);
				if(fs::exists(save_file)) {
					std::cout << std::format("skipping index: {}", save_file.string()) << std::endl;
				} else {
					std::cout << std::format("generating index: {}", save_file.string())
							  << std::endl;

					hnswlib::InnerProductSpace cosine_space(gist_vectors.dim);
					hnswlib::HierarchicalNSW<float> alg_hnsw = hnswlib::HierarchicalNSW<float>(
						&cosine_space, gist_vectors.nb, m, ef_construction);
					build_hnsw(alg_hnsw, gist_vectors, true);
					alg_hnsw.saveIndex(save_file.string());
					std::cout << std::endl;
				}
			}
		}
	}

	return 0;
}