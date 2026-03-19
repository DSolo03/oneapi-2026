#include "acc_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiAccONEAPI(const std::vector<float> &matrix_a,
                                   const std::vector<float> &vector_b,
                                   float accuracy, sycl::device device) {
  int matrix_size = vector_b.size();

  std::vector<float> current_solution(matrix_size, 0.0f);
  std::vector<float> next_solution(matrix_size, 0.0f);

  sycl::buffer<float, 1> matrix_buffer(matrix_a.data(),
                                       sycl::range<1>(matrix_a.size()));
  sycl::buffer<float, 1> rhs_buffer(vector_b.data(),
                                    sycl::range<1>(vector_b.size()));
  sycl::buffer<float, 1> current_buffer(current_solution.data(),
                                        sycl::range<1>(matrix_size));
  sycl::buffer<float, 1> next_buffer(next_solution.data(),
                                     sycl::range<1>(matrix_size));

  sycl::queue computation_queue(device);

  for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
    computation_queue
        .submit([&](sycl::handler &command_group) {
          auto matrix_accessor =
              matrix_buffer.get_access<sycl::access::mode::read>(command_group);
          auto rhs_accessor =
              rhs_buffer.get_access<sycl::access::mode::read>(command_group);
          auto current_accessor =
              current_buffer.get_access<sycl::access::mode::read>(
                  command_group);
          auto next_accessor =
              next_buffer.get_access<sycl::access::mode::write>(command_group);

          command_group.parallel_for<class jacobi_iteration>(
              sycl::range<1>(matrix_size), [=](sycl::id<1> element_id) {
                int row = element_id[0];
                float diagonal_element =
                    matrix_accessor[row * matrix_size + row];
                float summation = 0.0f;

                for (int column = 0; column < matrix_size; ++column) {
                  if (column != row) {
                    summation += matrix_accessor[row * matrix_size + column] *
                                 current_accessor[column];
                  }
                }

                next_accessor[row] =
                    (rhs_accessor[row] - summation) / diagonal_element;
              });
        })
        .wait();

    bool solution_converged = true;
    {
      auto next_host = next_buffer.get_host_access(sycl::read_only);
      auto current_host = current_buffer.get_host_access(sycl::write_only);

      for (int i = 0; i < matrix_size; ++i) {
        float difference = std::fabs(next_host[i] - current_host[i]);
        current_host[i] = next_host[i];

        if (difference >= accuracy) {
          solution_converged = false;
        }
      }
    }

    if (solution_converged) {
      break;
    }
  }

  std::vector<float> final_solution(matrix_size);
  {
    auto result_accessor = current_buffer.get_host_access(sycl::read_only);
    for (int i = 0; i < matrix_size; ++i) {
      final_solution[i] = result_accessor[i];
    }
  }

  return final_solution;
}