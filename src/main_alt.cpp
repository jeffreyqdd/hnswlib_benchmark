#include <hnswlib/hnswlib.h>
#include <thread>

const int dim = 960; // Dimension of the elements
const int nb = 100000; // Maximum number of elements, should be known beforehand
const int M = 32; // Tightly connected with internal dimensionality of the data
	// strongly affects the memory consumption
const int ef_construction = 200; // Controls index search speed/build speed tradeoff
const int num_threads = 20; // Number of threads for operations with index
int main() {
	return 0;
}
