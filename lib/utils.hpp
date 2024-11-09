#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
	if(numThreads <= 0) {
		numThreads = std::thread::hardware_concurrency();
	}

	if(numThreads == 1) {
		for(size_t id = start; id < end; id++) {
			fn(id, 0);
		}
	} else {
		std::vector<std::thread> threads;
		std::atomic<size_t> current(start);

		// keep track of exceptions in threads
		// https://stackoverflow.com/a/32428427/1713196
		std::exception_ptr lastException = nullptr;
		std::mutex lastExceptMutex;

		auto printer = [&end, &current]() {
			auto time_start = std::chrono::high_resolution_clock::now();
			while(true) {
				if(current >= end) {
					break;
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(1000));
				// std::cout << "\r" << current << "                  ";
				auto curr = current.load();
				double progress = static_cast<double>(curr) / end;
				auto time_now = std::chrono::high_resolution_clock::now();
				auto time_elapsed =
					std::chrono::duration_cast<std::chrono::seconds>(time_now - time_start).count();

				auto it_per_s = static_cast<double>(curr) / static_cast<double>(time_elapsed);
				auto remaining_time = static_cast<int>(static_cast<double>(end - curr) /
													   (static_cast<double>(it_per_s) + 0.001));

				std::cout << std::format("[{:7.2f}%] remaining time = {:5d}s, elapsed = {:5d}s  \r",
										 (progress * 100),
										 remaining_time,
										 time_elapsed);
				std::cout.flush();
			}

			std::cout << std::endl;
		};
		std::thread progress(printer);

		for(size_t threadId = 0; threadId < numThreads; ++threadId) {
			threads.push_back(std::thread([&, threadId] {
				while(true) {
					size_t id = current.fetch_add(1);

					if(id >= end) {
						break;
					}

					try {
						fn(id, threadId);
					} catch(...) {
						std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
						lastException = std::current_exception();
						/*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
						current = end;
						break;
					}
				}
			}));
		}

		progress.join();
		for(auto& thread : threads) {
			thread.join();
		}
		if(lastException) {
			std::rethrow_exception(lastException);
		}
	}
}