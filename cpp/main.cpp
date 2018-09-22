#include <array>
#include <tuple>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <type_traits>
#include <vector>
#include <chrono>
#include <numeric>

//////////////////////////////////////////////////////////////////////////
// Helper Stuff
//////////////////////////////////////////////////////////////////////////

template <class T>
void print_container(T& container) {
    for (auto el : container)
        std::cout << el << ", ";
    std::cout << "\n";
}

template <typename T, typename F, std::size_t... I>
void for_each_impl(F&& f, T&& tuple, std::index_sequence<I...>) {
    (void) std::initializer_list<int>{
        (f(std::get<I>(std::forward<T>(tuple))), void(), int{})...
    };
}

template <typename T, typename F>
void for_each(F&& f, T&& tuple) {
    constexpr std::size_t N = std::tuple_size<std::decay_t<T>>::value;
    for_each_impl(std::forward<F>(f), std::forward<T>(tuple),
                  std::make_index_sequence<N>{});
}

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    size_t elapsed() const { 
        return std::chrono::duration_cast<std::chrono::nanoseconds>
            (clock_::now() - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    std::chrono::time_point<clock_> beg_;
};

//////////////////////////////////////////////////////////////////////////
// Array class
//////////////////////////////////////////////////////////////////////////

enum layout { row_major, col_major };

template <std::size_t N, std::size_t I = 0, class... X, class... Args>
auto recursive_for(const std::tuple<X...>& x, Args... args) {
    if constexpr (I == N)
        for (std::size_t i = 0; i < std::get<0>(x).shape[I]; ++i)
            std::get<0>(x)(args..., i);
    else
        for (std::size_t i = 0; i < std::get<0>(x).shape[I]; ++i) {
            if constexpr (sizeof...(X) > N)
                std::get<N>(x)(args..., i);
            recursive_for<N, I + 1>(x, args..., i);
        }
}

template <class T, layout L = row_major>
struct qiterator
{
    using index_type = typename T::shape_type;

    qiterator(const T& parent)
        : m_ref(parent), m_size(m_ref.size()), m_index{0} {
    }

    qiterator& operator++() {
        m_linear_idx++;
        if constexpr (L == row_major)
        {
            for (std::size_t i = std::tuple_size_v<index_type>; ++m_index[i - 1], i > 0; --i)
            {
                if (m_index[i - 1] == m_ref.shape[i - 1]) { m_index[i - 1] = 0; }
                else { return *this; }
            }
        }
        else 
        {
            for (std::size_t i = 0; ++m_index[i], i < std::tuple_size_v<index_type>; ++i)
            {
                if (m_index[i] == m_ref.shape[i]) { m_index[i] = 0; }
                else { return *this; }
            }
        }
        return *this;
    }

    template <std::size_t... I>
    auto deref_impl(std::index_sequence<I...>) { return m_ref(std::get<I>(m_index)...); }
    template <std::size_t... I>
    const auto deref_impl(std::index_sequence<I...>) const { return m_ref(std::get<I>(m_index)...); }

    auto operator*() { return deref_impl(std::make_index_sequence<std::tuple_size_v<index_type>>{}); }
    const auto operator*() const { return deref_impl(std::make_index_sequence<std::tuple_size_v<index_type>>{}); }

    bool operator==(const qiterator&) { return m_linear_idx == m_size; }
    bool operator!=(const qiterator& end) { return !(*this == end); }

    const T& m_ref;
    index_type m_index;
    std::size_t m_linear_idx = 0, m_size = 0;
};

template <class T>
void container_resize(T& container, std::size_t sz) { container.resize(sz); };
template <class T, std::size_t N>
void container_resize(std::array<T, N>& container, std::size_t sz) {};

template <class L, class... X>
class qfunction;

template <class CT, std::size_t N, layout L = row_major>
class simple_array_view {
public:

    using self_type = simple_array_view<CT, N>;
    using shape_type = std::array<ptrdiff_t, N>;
    using container_type = std::decay_t<CT>;
    using value_type = std::decay_t<decltype(std::declval<CT>()[0])>;
    using container_reference = std::decay_t<CT>&;

    simple_array_view() = default;

    auto compute_strides()
    {
        ptrdiff_t data_size = 1;
        if constexpr (N > 0) {
            if constexpr (L == row_major) {
                strides[N - 1] = shape[N - 1] != 1 ? 1 : 0;
                for (std::ptrdiff_t i = N - 1; i > 0; --i) {
                    data_size *= static_cast<ptrdiff_t>(shape[i]);
                    strides[i - 1] = shape[i - 1] != 1 ? data_size : 0;
                }
                data_size *= shape[0];
            }
            else {
                for (std::size_t i = 0; i < N; ++i) {
                    strides[i] = data_size;
                    data_size = strides[i] * static_cast<ptrdiff_t>(shape[i]);
                    if (shape[i] == 1) { strides[i] = 0; }
                }
            }
        }
        return data_size;
    }

    auto constexpr compute_offset() const { return ptrdiff_t(0); }; 

    template <class Arg, class... Args>
    auto constexpr compute_offset(Arg a1, Args... args) const {
        if constexpr (sizeof...(Args) + 1 > N)
            return compute_offset(args...);
        else {
            std::array<ptrdiff_t, sizeof...(Args) + 1> idx({static_cast<long>(a1), static_cast<long>(args)...});
            ptrdiff_t offset = 0;
            for (std::size_t i = 0; i < N; ++i) {
                offset += strides[i] * idx[i];
            }
            return offset;
        }
    }

    explicit simple_array_view(value_type data, const std::array<ptrdiff_t, N>& i_shape) : shape(i_shape) {
        container_resize(memory, compute_strides());
        std::fill(memory.begin(), memory.end(), data);
        compute_strides();
    }

    explicit simple_array_view(CT data, const std::array<ptrdiff_t, N>& i_shape,
                               const std::array<ptrdiff_t, N>& i_strides) : memory(data), shape(i_shape), strides(i_strides)
    {
    }

    template <class T>
    explicit simple_array_view(T&& data, const std::array<ptrdiff_t, N>& i_shape) : memory(std::forward<T>(data)), shape(i_shape) {
        compute_strides();
    }

    explicit simple_array_view(const std::array<ptrdiff_t, N>& i_shape) : shape(i_shape) {
        container_resize(memory, compute_strides());
    }

    template <class T>
    void assign_impl(T&& rhs) {
        auto assign_func = make_qfunc([](auto& lhs, auto rhs) { lhs = rhs; }, *this, rhs);
        recursive_for<std::tuple_size<shape_type>::value - 1>(std::make_tuple(std::move(assign_func)));
    }

    template <class LM, class... X>
    simple_array_view(const qfunction<LM, X...>& e) : shape(e.shape) {
        container_resize(memory, compute_strides());
        assign_impl(e);
    }

    template <class LM, class... X>
    self_type& operator=(const qfunction<LM, X...>& e) {
        if (!std::equal(shape.begin(), shape.end(), e.shape.begin())) {
            std::copy(e.shape.begin(), e.shape.end(), shape.begin());
            container_resize(memory, compute_strides());
        }
        assign_impl(e);
        return *this;
    }

    auto begin() { return qiterator(*this); }
    auto end() { return qiterator(*this); }

    void fill(value_type val) { std::fill(memory.begin(), memory.end(), val); }

    template <class... Args>
    value_type& operator()(Args... args) { return memory[compute_offset(args...)]; }

    template <class... Args>
    const value_type& operator()(Args... args) const { return memory[compute_offset(args...)]; }

    container_reference data() { return memory; }
    size_t size() const { return memory.size(); };

    CT memory;
    std::array<ptrdiff_t, N> shape;
    std::array<ptrdiff_t, N> strides;
};

template <class... Args>
constexpr auto max_dim() {
    constexpr auto arr = std::array<size_t, sizeof...(Args)>{std::tuple_size<Args>::value...};
    return *std::max_element(arr.begin(), arr.end());
}

////////////////////////////////////////////////////////////////////////////////
// Lazy function
////////////////////////////////////////////////////////////////////////////////

template <class F, class... X>
class qfunction
{
public:

    using shape_type = std::array<ptrdiff_t, max_dim<typename std::decay_t<X>::shape_type...>()>;

    template <class... Args>
    qfunction(F f, Args&&... args)
        : m_f(f), m_args(std::forward<Args>(args)...)
    {
        std::fill(shape.begin(), shape.end(), 1);

        auto broadcast_shape = [this](const auto& v) constexpr {
            std::size_t offset = this->shape.size() - v.shape.size();
            for (std::size_t i = 0; i < v.shape.size(); ++i) {
                if (this->shape[offset + i] == 1)
                    this->shape[offset + i] = v.shape[i];
                else
                    if (v.shape[i] != this->shape[offset + i] && v.shape[i] != 1)
                        throw std::runtime_error("Broadcast error.");
            }
            return true;
        };

        for_each(broadcast_shape, m_args);
    }

    template <std::size_t... I, class... Args>
    auto access_impl(std::index_sequence<I...>, Args... args) const { 
        return m_f(std::get<I>(m_args)(args...)...); 
    }

    template <class... Args>
    auto operator()(Args... args) const {
        return access_impl(std::make_index_sequence<sizeof...(X)>(), args...);
    }

    auto begin() { return qiterator(*this); }
    auto end() { return qiterator(*this); }
    auto begin() const { return qiterator(*this); }
    auto end() const { return qiterator(*this); }

    size_t size() const { return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>{}); }

    F m_f;
    shape_type shape;
    std::tuple<X...> m_args;
};

template <std::size_t N, class T = double>
using qarray = simple_array_view<std::vector<T>, N>;

template <std::size_t N, class T = double>
using qview = simple_array_view<T*, N>;

template <class T>
decltype(auto) wrap_if_scalar(T&& arg)
{
    if constexpr (std::is_scalar_v<std::decay_t<decltype(arg)>>)
        return qarray<0>({arg}, {});
    else
        return std::forward<T>(arg);
}

template <class S>
struct closure_type {
    using underlying_type = std::conditional_t<std::is_const<std::remove_reference_t<S>>::value,
                                               const std::decay_t<S>,
                                               std::decay_t<S>>;
    using type = typename std::conditional<std::is_lvalue_reference<S>::value,
                                           underlying_type&,
                                           underlying_type>::type;
};

template <class T>
using closure_type_t = typename closure_type<T>::type;

template <class L, class... Args>
auto make_qfunc(L func, Args&&... args) {
    return qfunction<L, closure_type_t<Args>...>(func, std::forward<Args>(args)...);
}

#define BINARY_OP(OP)                                                                   \
    template <class A, class B>                                                         \
    auto operator OP (A&& a, B&& b) {                                                   \
        return make_qfunc([](auto x, auto y) { return x OP y; },                        \
            wrap_if_scalar(std::forward<A>(a)), wrap_if_scalar(std::forward<B>(b)));    \
    }

#define UNARY_FUNC(FUNC)                                                                   \
    template <class A>                                                                     \
    auto FUNC (A&& a) {                                                                    \
        return make_qfunc([](auto x) {                                                     \
            using namespace std; return FUNC (x); }, wrap_if_scalar(std::forward<A>(a)));  \
    }

BINARY_OP(+); BINARY_OP(-); BINARY_OP(*); BINARY_OP(/);
UNARY_FUNC(sin); UNARY_FUNC(cos);

#include <iomanip>
template <class T, class O = std::ostream>
void qprint(T&& t, O& os = std::cout)
{
    using shape_type = typename std::decay_t<T>::shape_type;
    auto print_func = make_qfunc([&os](auto el) { 
        if constexpr (std::is_scalar_v<decltype(el)>) os << std::fixed << std::setprecision(2) << std::setw(8) << el << ", ";
        else qprint(el);
    },
    std::forward<T>(t));
    recursive_for<std::tuple_size<shape_type>::value - 1>(std::make_tuple(print_func, [&os](auto) { os << "\n"; }));
    os << "\n";
}

using Point3 = simple_array_view<std::array<float, 3>, 1>;

template <class C>
constexpr auto compute_size(const C& cnt)
{
    return std::accumulate(cnt.cbegin(), cnt.cend(), size_t(1), std::multiplies<size_t>());
}

template <class T, class I, std::size_t N>
auto sum_axis(const T& t, const I (&axes)[N])
{
    auto res_shape = t.shape;
    for (auto el : axes)
        res_shape[el] = 1;
    auto data = std::vector<float>(compute_size(res_shape));
    qarray<2, float> res(data, res_shape);

    auto sum_func = make_qfunc([](auto& lhs, auto rhs) {
         lhs += rhs;
    }, res, t);
    recursive_for<std::tuple_size<typename T::shape_type>::value - 1>(std::make_tuple(sum_func));
    return res;
}

// auto sum(const Point3& a)
// {
//     float result = 0;
//     for (std::size_t i = 0; i < a.size(); ++i)
//     {
//         result += a.memory[i];
//     }
//     return result;
// }

template <class T>
auto cumsum(const T& t, std::ptrdiff_t ax)
{
    std::array<ptrdiff_t, 2> index = {0, 0};
    index[ax] = 1; // first elem
    auto offset = std::inner_product(index.begin(), index.end(), t.strides.begin(), 0);

    auto view_shape = t.shape;
    view_shape[ax] -= 1;

    qarray<2, double> res = t; // copy (copy only first "line")

    // pick first element, e.g. res(0, 1) but indexing with std array not possible r
    qview<2, double> rhs_v(&res(0, 0), view_shape, t.strides);
    qview<2, double> lhs_v(&res.memory[offset], view_shape, t.strides);
    auto sum_func = make_qfunc([](auto& lhs, auto rhs) {
         lhs += rhs;
    }, lhs_v, rhs_v);

    recursive_for<std::tuple_size<typename T::shape_type>::value - 1>(std::make_tuple(sum_func));
    return res;
}


template <class T>
auto sum(const T& t)
{
    return std::accumulate(t.begin(), t.end(), 0.f);
}

auto super_custom_func(const Point3& a, const Point3& b)
{
    return std::sqrt(sum(a * b));
}

int main()
{
    // Just to show that broadcasting works
    auto b1 = std::vector<double>(25);
    std::iota(b1.begin(), b1.end(), 0.0);
    auto b2 = std::vector<double>(5);
    std::iota(b2.begin(), b2.end(), 0.0);
    auto b3 = 2.5;
    auto bc_func = qarray<2>(b1, {5, 5}) + qarray<1>(b2, {5})  * b3;
    qprint(bc_func);

    auto q1 = qarray<2>(b1, {5, 5});

    qprint(cumsum(q1, 0));
    qprint(cumsum(q1, 1));

    // std::cout << sum(bc_func) << std::endl;

    auto d1 = std::vector<double>(1000 * 1000, 0.1);
    auto d2 = std::vector<double>(1000, 0.232);
    auto dres = std::vector<double>(1000 * 1000, 0);
    double c = 1.0;

    simple_array_view<std::vector<double>, 2> a(d1, {1000, 1000});
    simple_array_view<std::vector<double>, 1> b(d2, {1000});
    simple_array_view<std::vector<double>, 2> res(dres, {1000, 1000});

    auto sqfunc = a + b - sin(c);
    
    std::size_t min_time = 100000000;

    qprint(cumsum(q1, 1));
    std::cout << "\n\nBenchmarking cumsum(a[5x5], 1)\n===============================\n";

    for (int i = 0; i < 10'000; ++i)
    {
        Timer timer;
        res = cumsum(q1, 1);
        auto elapsed = timer.elapsed(); // nanoseconds
        if (elapsed < min_time) min_time = elapsed;
        // std::cout << "TIME: "  << (double) elapsed / (double) 1000 << " μs" << std::endl;
    }

    std::cout << "\nMIN TIME: " << min_time << " ns\n";
    std::cout << "        = " << (double) min_time / (double) 1000 << " μs" << std::endl;

    min_time = 100000000;

    std::cout << "\n\nBenchmarking sum_axis(a[5x5], {0})\n===============================\n";

    for (int i = 0; i < 10'000; ++i)
    {
        Timer timer;
        auto s_res = sum_axis(q1, {0});
        auto elapsed = timer.elapsed(); // nanoseconds
        if (elapsed < min_time) min_time = elapsed;
        // std::cout << "TIME: "  << (double) elapsed / (double) 1000 << " μs" << std::endl;
    }

    std::cout << "\nMIN TIME: " << min_time << " ns\n";
    std::cout << "        = " << (double) min_time / (double) 1000 << " μs" << std::endl;

    min_time = 100000000;
    std::cout << "\n\nBenchmarking a[1000x1000] + b[1000] - sin(c[])\n===================================\n";
    for (int i = 0; i < 200; ++i)
    {
        Timer timer;
        res = sqfunc;
        auto elapsed = timer.elapsed(); // nanoseconds
        if (elapsed < min_time) min_time = elapsed;
        // std::cout << "TIME: "  << (double) elapsed / (double) 1000 << " μs" << std::endl;
    }

    std::cout << "\nMIN TIME: " << min_time << " ns\n";
    std::cout << "        = " << (double) min_time / (double) 1000 << " μs" << std::endl;


    constexpr std::ptrdiff_t psz = 1'000'000;

    simple_array_view<std::vector<Point3>, 1> px(std::vector<Point3>(psz, Point3(std::array<float, 3>{0.5, 2.1, 3.2}, {3})), {psz});
    simple_array_view<std::vector<Point3>, 1> py(std::vector<Point3>(psz, Point3(std::array<float, 3>{0.5, 2.1, 3.2}, {3})), {psz});
    simple_array_view<std::vector<float>, 1> pres(std::vector<float>(psz), {psz});

    auto pointfunc = make_qfunc(super_custom_func, px, py);

    pres = make_qfunc(super_custom_func, px, py);
    auto bc_super_custom_func = make_qfunc(super_custom_func, px, py);

    min_time = 100000000;

    std::cout << "\n\nBenchmarking super_custom_func\n===============================\n";

    for (int i = 0; i < 200; ++i)
    {
        Timer timer;
        pres = bc_super_custom_func;
        auto elapsed = timer.elapsed(); // nanoseconds
        if (elapsed < min_time) min_time = elapsed;
        // std::cout << "TIME: "  << (double) elapsed / (double) 1000 << " μs" << std::endl;
    }

    std::cout << "\nMIN TIME: " << min_time << " ns\n";
    std::cout << "        = " << (double) min_time / (double) 1000 << " μs\n" << std::endl;
}