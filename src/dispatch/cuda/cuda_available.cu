#include <cuda_runtime.h>

#include <propr/runtime/dispatch.hpp>

using namespace propr::runtime;


#ifdef PROPR_HAS_CUDA

    bool cuda_is_available() {
        static int cached = -1;
        if (cached < 0) {
            int device_count = 0;
            const cudaError_t status = cudaGetDeviceCount(&device_count);
            if (status != cudaSuccess) {
                cudaGetLastError();
                cached = 0;
            } else {
                cached = device_count > 0 ? 1 : 0;
            }
        }
        return cached > 0;
    }

#endif
