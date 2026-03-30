#include "dev_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiDevONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    float accuracy, sycl::device device) {

    size_t n = b.size();

    if (n == 0 || a.size() != n * n) {
        return std::vector<float>();
    }

    sycl::queue queue(device);

    float* d_a = sycl::malloc_device<float>(a.size(), queue);
    float* d_b = sycl::malloc_device<float>(b.size(), queue);
    float* d_x = sycl::malloc_device<float>(n, queue);
    float* d_x_new = sycl::malloc_device<float>(n, queue);

    queue.memcpy(d_a, a.data(), sizeof(float) * a.size());
    queue.memcpy(d_b, b.data(), sizeof(float) * b.size());

    queue.memset(d_x, 0, sizeof(float) * n);

    queue.wait();

    std::vector<float> x(n);

    bool converged = false;
    int iteration = 0;

    while (!converged && iteration < ITERATIONS) {
        queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            size_t row = i[0];
            float sum = 0.0f;
            float diag = d_a[row * n + row];

            for (size_t j = 0; j < n; j++) {
                if (j != row) {
                    sum += d_a[row * n + j] * d_x[j];
                }
            }

            d_x_new[row] = (d_b[row] - sum) / diag;
            });

        queue.wait();

        std::vector<float> x_new(n);
        queue.memcpy(x_new.data(), d_x_new, sizeof(float) * n).wait();
        queue.memcpy(x.data(), d_x, sizeof(float) * n).wait();

        converged = true;

        for (size_t i = 0; i < n; i++) {
            float diff = std::fabs(x_new[i] - x[i]);

            if (diff >= accuracy) {
                converged = false;
            }

            x[i] = x_new[i];
        }

        queue.memcpy(d_x, x.data(), sizeof(float) * n).wait();

        iteration++;
    }

    queue.memcpy(x.data(), d_x, sizeof(float) * n).wait();

    sycl::free(d_a, queue);
    sycl::free(d_b, queue);
    sycl::free(d_x, queue);
    sycl::free(d_x_new, queue);

    return x;
}
