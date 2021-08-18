#include <atomic>

typedef struct {
  int gridSize;    // Number of thread blocks per grid
  int blockSize;   // Number of threads per thread block
  int smemSize; // Shared Memory Size
  int stream;      // associated stream
} launchConfig;


static int *kernels = nullptr;
static std::atomic<unsigned long> num_kernels = {0};
static std::atomic<unsigned long> synced_kernels = {0};

/// Kernel launch
template <typename Ty, typename Func, Func kernel, typename... Args>
void launch(const launchConfig &config, Ty *ptrA, Ty *ptrB, Args... args) {

  int kernel_no = num_kernels++;
#pragma omp target teams is_device_ptr(ptrA, ptrB) num_teams(config.gridSize)     \
    thread_limit(config.blockSize) depend(out                                     \
                                       : kernels[kernel_no]) nowait
  {

#pragma omp parallel
    { kernel(ptrA, ptrB, args...); }
  }
}


/// Device Synchronization
void synchronize() {
  unsigned long kernel_first = synced_kernels;
  unsigned long kernel_last = num_kernels;
  if (kernel_first < kernel_last) {
    for (unsigned long i = kernel_first; i < kernel_last; ++i) {
#pragma omp task if(0) depend(in : kernels[i])
      {}
    }
    synced_kernels.compare_exchange_strong(kernel_first, kernel_last);
  }
}


