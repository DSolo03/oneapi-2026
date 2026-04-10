#include "shared_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiSharedONEAPI(const std::vector<float> &a, const std::vector<float> &b, float accuracy,
                                      sycl::device device)
{
    const int n = static_cast<int>(b.size());

    sycl::queue queue(device);

    float *matrix = sycl::malloc_shared<float>(n * n, queue);
    float *rhs = sycl::malloc_shared<float>(n, queue);
    float *previous = sycl::malloc_shared<float>(n, queue);
    float *current = sycl::malloc_shared<float>(n, queue);

    for (int i = 0; i < n * n; ++i)
    {
        matrix[i] = a[i];
    }

    for (int i = 0; i < n; ++i)
    {
        rhs[i] = b[i];
        previous[i] = 0.0f;
        current[i] = 0.0f;
    }

    bool converged = false;

    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        queue
            .parallel_for(sycl::range<1>(n),
                          [=](sycl::id<1> idx) {
                              const int row = static_cast<int>(idx[0]);

                              float sum = 0.0f;
                              for (int col = 0; col < n; ++col)
                              {
                                  if (col != row)
                                  {
                                      sum += matrix[row * n + col] * previous[col];
                                  }
                              }

                              current[row] = (rhs[row] - sum) / matrix[row * n + row];
                          })
            .wait();

        float max_diff = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            const float diff = std::fabs(current[i] - previous[i]);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
        }

        if (max_diff < accuracy)
        {
            converged = true;
            break;
        }

        float *tmp = previous;
        previous = current;
        current = tmp;
    }

    const float *answer = converged ? current : previous;

    std::vector<float> result(n);
    for (int i = 0; i < n; ++i)
    {
        result[i] = answer[i];
    }

    sycl::free(matrix, queue);
    sycl::free(rhs, queue);
    sycl::free(previous, queue);
    sycl::free(current, queue);

    return result;
}
