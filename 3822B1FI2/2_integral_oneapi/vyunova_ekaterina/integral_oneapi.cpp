#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
  float result = 0.0f;
  const float step = (end - start) / count;

  sycl::queue q(device);

  {
    sycl::buffer<float> result_buf(&result, 1);

    q.submit([&](sycl::handler& cgh) {
      auto sum = sycl::reduction(result_buf, cgh, sycl::plus<float>());

      cgh.parallel_for(
          sycl::range<2>(count, count), sum,
          [=](sycl::id<2> id, auto& total) {
            float x = start + step * (static_cast<float>(id[0]) + 0.5f);
            float y = start + step * (static_cast<float>(id[1]) + 0.5f);
            total += sycl::sin(x) * sycl::cos(y);
          });
    }).wait();
  }

  return result * step * step;
}
