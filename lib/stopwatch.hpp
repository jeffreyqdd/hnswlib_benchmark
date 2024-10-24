#include <chrono>

namespace chrono = std::chrono;

class StopWatch {
public:
	StopWatch() {
		_time_begin = chrono::steady_clock::now();
	}

	inline void start() {
		_time_begin = chrono::steady_clock::now();
	}

	inline uint64_t end() {
		chrono::steady_clock::time_point time_end = chrono::steady_clock::now();
		auto delta = time_end - _time_begin;
		return chrono::duration_cast<chrono::microseconds>(delta).count();
	}

private:
	chrono::steady_clock::time_point _time_begin;
};