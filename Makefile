build-release:
	./build.sh Release

benchmark: build-release
	./build-Release/bench_st -p ./results

.PHONY: benchmark