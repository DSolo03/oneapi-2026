#include "acc_jacobi_oneapi.h"

#include <utility>

std::vector<float> JacobiAccONEAPI(const std::vector<float> &a, const std::vector<float> &b, float accuracy,
                                   sycl::device device)
{
    const size_t n = b.size();

    std::vector<float> previous(n, 0.0f);
    std::vector<float> current(n, 0.0f);
    std::vector<float> diagonal(n);

    for (size_t i = 0; i < n; ++i)
    {
        diagonal[i] = a[i * n + i];
    }

    sycl::queue queue(device);

    sycl::buffer a_buffer(a.data(), sycl::range<1>(n * n));
    sycl::buffer b_buffer(b.data(), sycl::range<1>(n));
    sycl::buffer diag_buffer(diagonal.data(), sycl::range<1>(n));

    int iterations_left = ITERATIONS;
    float max_difference = 0.0f;

    do
    {
        max_difference = 0.0f;
        std::swap(previous, current);

        sycl::buffer prev_buffer(previous.data(), sycl::range<1>(n));
        sycl::buffer curr_buffer(current.data(), sycl::range<1>(n));
        sycl::buffer diff_buffer(&max_difference, sycl::range<1>(1));

        queue
            .submit([&](sycl::handler &cgh) {
                auto matrix = a_buffer.get_access<sycl::access::mode::read>(cgh);
                auto rhs = b_buffer.get_access<sycl::access::mode::read>(cgh);
                auto diag = diag_buffer.get_access<sycl::access::mode::read>(cgh);
                auto prev = prev_buffer.get_access<sycl::access::mode::read>(cgh);
                auto curr = curr_buffer.get_access<sycl::access::mode::write>(cgh);

                auto diff_reduction = sycl::reduction(diff_buffer, cgh, sycl::maximum<float>());

                cgh.parallel_for(sycl::range<1>(n), diff_reduction, [=](sycl::id<1> idx, auto &diff_max) {
                    const size_t row = idx[0];
                    const size_t offset = row * n;

                    float sum = 0.0f;
                    for (size_t col = 0; col < n; ++col)
                    {
                        sum += matrix[offset + col] * prev[col];
                    }

                    sum -= diag[row] * prev[row];

                    const float value = (rhs[row] - sum) / diag[row];
                    curr[row] = value;

                    diff_max.combine(sycl::fabs(value - prev[row]));
                });
            })
            .wait();

        --iterations_left;
    } while (max_difference >= accuracy && iterations_left > 0);

    return current;
}
