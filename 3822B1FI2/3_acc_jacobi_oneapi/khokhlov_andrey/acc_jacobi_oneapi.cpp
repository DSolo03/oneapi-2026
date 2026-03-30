#include "acc_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

using buffer_t = sycl::buffer<float, 1>;

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    const size_t matrix_size = b.size();
    std::vector<float> current_solution(matrix_size, 0.0f);
    std::vector<float> next_solution(matrix_size, 0.0f);
    if (matrix_size == 0 || a.size() != matrix_size * matrix_size) {
        return {};
    }

    sycl::queue compute_queue(device);
    
    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        buffer_t matrix_buffer(a.data(), sycl::range<1>(a.size()));
        buffer_t rhs_buffer(b.data(), sycl::range<1>(matrix_size));
        buffer_t current_buffer(current_solution.data(), sycl::range<1>(matrix_size));
        buffer_t next_buffer(next_solution.data(), sycl::range<1>(matrix_size));

        float max_difference = 0.0f;
        buffer_t diff_buffer(&max_difference, sycl::range<1>(1));
        
        compute_queue.submit([&](sycl::handler& kernel_handler) {
            auto matrix_access = matrix_buffer.get_access<sycl::access::mode::read>(kernel_handler);
            auto rhs_access = rhs_buffer.get_access<sycl::access::mode::read>(kernel_handler);
            auto current_access = current_buffer.get_access<sycl::access::mode::read>(kernel_handler);
            auto next_access = next_buffer.get_access<sycl::access::mode::write>(kernel_handler);
            auto reduction_max = sycl::reduction(diff_buffer, kernel_handler, sycl::maximum<float>());
            kernel_handler.parallel_for(
                sycl::range<1>(matrix_size),
                reduction_max,
                [=](sycl::id<1> element_id, auto& max_value) {
                    const size_t row_index = element_id[0];
                    float sum_off_diagonal = 0.0f;
                    
                    for (size_t col_index = 0; col_index < matrix_size; ++col_index) {
                        if (col_index != row_index) {
                            sum_off_diagonal += matrix_access[row_index * matrix_size + col_index] * 
                                               current_access[col_index];
                        }
                    }
                    float diagonal_element = matrix_access[row_index * matrix_size + row_index];
                    float new_value = (rhs_access[row_index] - sum_off_diagonal) / diagonal_element;
                    next_access[row_index] = new_value;
                    float difference = sycl::fabs(new_value - current_access[row_index]);
                    max_value.combine(difference);
                }
            );
        }).wait();
        {
            sycl::host_accessor diff_host_accessor(diff_buffer, sycl::read_only);
            max_difference = diff_host_accessor[0];
        }
        if (max_difference < accuracy) {
            break;
        }
        std::swap(current_solution, next_solution);
    }
    
    return current_solution;
}