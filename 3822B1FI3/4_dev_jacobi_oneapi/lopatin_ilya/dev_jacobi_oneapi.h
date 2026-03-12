#ifndef __JACOBI_DEV_ONEAPI_H
#define __JACOBI_DEV_ONEAPI_H

#include <vector>

#include <sycl/sycl.hpp>

#ifndef ITERATIONS
#define ITERATIONS 1024
#endif // !ITERATIONS

std::vector<float> JacobiDevONEAPI(
    const std::vector<float> a, const std::vector<float> b,
    float accuracy, sycl::device device);

#endif  // __JACOBI_DEV_ONEAPI_H