#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>

#include <xtensor/xfixed.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xvectorize.hpp>

// Simple stopwatch object
class stopwatch
{
public:

    stopwatch() : m_start(clock_type::now())
    {
    }

    void reset()
    {
        m_start = clock_type::now();
    }

    std::size_t elapsed() const
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now() - m_start).count();
    }

private:

    typedef std::chrono::high_resolution_clock clock_type;
    std::chrono::time_point<clock_type> m_start;
};

template <class T>
using xpoint = xt::xtensor_fixed<T, xt::xshape<3>>;

template <class T>
auto sum_xpoint(const xpoint<T>& t)
{
    return std::accumulate(t.cbegin(), t.cend(), T());
}

// Simple test program
int main()
{
    // First Benchmark:
    //
    //  a[1000x1000] + b[1000] - sin(c[])
    {
        xt::xtensor<double, 2> a = xt::random::rand<double>({1000, 1000});
        xt::xtensor<double, 1> b = xt::random::rand<double>({1000});
        double c = 1.0;

        // Un-evaluated broadcasting exprression
        auto expr = a + b - std::sin(c);
        auto res = xt::xtensor<double, 2>::from_shape({1000, 1000});

        // Benchmark loop
        std::cout << "Benchmarking a[1000x1000] + b[1000] - sin(c[])" << std::endl;
        std::size_t min_time = 100000000;
        for (int i = 0; i < 200; ++i)
        {
            stopwatch timer;                // Create timer
            xt::noalias(res) = expr;        // Evaluate the expression.
            auto elapsed = timer.elapsed(); // Nanoseconds
            if (elapsed < min_time)
            {
                min_time = elapsed;
            }
        }

        // Output results
        std::cout << "MIN TIME: " << min_time << " ns" << std::endl;
        std::cout << "        = " << (double) min_time / (double) 1000 << " μs" << std::endl;
        std::cout << std::endl << std::endl;
    }

    // Second Benchmark:
    //
    //  std::sqrt(sum(a * b));
    {
        constexpr std::size_t psz = 1000000;
        auto px = xt::xtensor<xpoint<float>, 1>({psz}, {0.5f, 2.1f, 3.2f}),
             py = xt::xtensor<xpoint<float>, 1>({psz}, {0.5f, 2.1f, 3.2f});
        auto res = xt::xtensor<float, 1>({psz});
        auto sum = xt::vectorize(sum_xpoint<float>);

        // Un-evaluated broadcasting expression
        auto expr = xt::sqrt(sum(px * py));

        // Benchmark loop
        std::cout << "Benchmarking sqrt(sum(a * b))" << std::endl;
        std::size_t min_time = 100000000;
        for (int i = 0; i < 200; ++i)
        {
            stopwatch timer;                // Create timer
            xt::noalias(res) = expr;        // Evaluate the expression.
            auto elapsed = timer.elapsed(); // Nanoseconds
            if (elapsed < min_time)
            {
                min_time = elapsed;
            }
        }

        // Output results
        std::cout << "MIN TIME: " << min_time << " ns" << std::endl;
        std::cout << "        = " << (double) min_time / (double) 1000 << " μs" << std::endl;
        std::cout << std::endl << std::endl;
    }
}
