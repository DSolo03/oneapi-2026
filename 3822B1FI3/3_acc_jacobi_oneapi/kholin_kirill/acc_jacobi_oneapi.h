#ifndef __ACC_JACOBI_ONEAPI_H
#define __ACC_JACOBI_ONEAPI_H

#include <sycl/sycl.hpp>
#include <vector>

#define MAX_ITERATIONS 1024

std::vector<float> JacobiAccONEAPI(const std::vector<float> &matrix_a,
                                   const std::vector<float> &vector_b,
                                   float accuracy, sycl::device device);

#endif // __ACC_JACOBI_ONEAPI_H