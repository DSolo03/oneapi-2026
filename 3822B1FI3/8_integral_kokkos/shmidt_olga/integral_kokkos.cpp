#include "integral_kokkos.h"
#include <cmath>

float IntegralKokkos(float start, float end, int count)
{
    Kokkos::initialize();

    float step = (end - start) / count;

    Kokkos::View<float> total("total");
    Kokkos::deep_copy(total, 0.0f);

    Kokkos::parallel_for(
        "integral",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { count, count }),
        KOKKOS_LAMBDA(int i, int j)
    {
        float x = start + (i + 0.5f) * step;
        float y = start + (j + 0.5f) * step;
        float val = Kokkos::sin(x) * Kokkos::cos(y) * step * step;

        Kokkos::atomic_add(&total(), val);
    });

    Kokkos::fence();

    auto host_total = Kokkos::create_mirror_view(total);
    Kokkos::deep_copy(host_total, total);

    float result = host_total();

    Kokkos::finalize();

    return result;
}