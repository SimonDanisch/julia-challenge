#include <array>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>

// type trait to check if the given type is considered a container or a value
// you're a container if std::size(x) and x[i] make sense
template<class T>
struct is_container {

  template <class F>
  static auto check(F x) -> decltype((x[0], std::size(x), std::true_type()));
  static std::false_type check(...);

  static constexpr bool value = decltype(check(std::declval<T>()))::value;
};

// utilities to compute depth, size and index of a random type
// this is not intrusive and work for any type
template <class... Ts> struct depth;
template <class T> struct depth<T> {
  static constexpr size_t get() {
    if constexpr (is_container<T>::value) return 1 + depth<decltype(std::declval<T>()[0])>::value;
    else return 0;
  }
  static constexpr size_t value = get();
};
template<class T0, class T1, class... Ts>
struct depth<T0, T1, Ts...> {
  static constexpr size_t value = std::max(depth<T0>::value, depth<T1, Ts...>::value);
};


template<class T>
constexpr size_t size(T const& x) {
  if constexpr (is_container<T>::value) return std::size(x);
  else return 1;
}

template<class T0, class T1, class... Ts>
constexpr size_t size(T0 const& x0, T1 const& x1, Ts... xs) {
  return std::max(::size(x0), ::size(x1, xs...));
}

template<class T>
constexpr decltype(auto) index(size_t i, T&& x) {
  if constexpr (is_container<T>::value) return std::forward<T>(x)[i];
  else return (std::forward<T>(x));
}

// lazy evaluation engine, with broadcasting
template<class F>
struct lazy {
  F f;

  template<class T, class... Args>
  void operator()(T& out, Args const&... args) const {
    if constexpr (depth<Args...>::value == 0) {
      f(out, args...);
    }
    else {
      size_t n = ::size(args...);
      for(size_t i = 0; i < n; ++i)
        operator()(index(i, out), index(i, args)...);
    }
  }

};

template<class F>
lazy<F> make_lazy(F f) {
  return {f};
}

// example etc

int main(int argc, char **argv) {
  auto op = make_lazy([](auto& out, auto x, auto y, auto z) { out = x + y - std::sin(z); });

  /* ndarrays, note how the types only fulfill the is_container trait */
  std::array<std::array<double, 10>, 10> x;
  for(auto & vi : x)
    for(auto& vj : vi)
      vj = 1;

  std::vector<double> y(10);
  std::iota(y.begin(), y.end(), 10);

  double z = 3.;

  std::array<std::array<double, 10>, 10> out;
  op(out, x, y, z);
  for(auto const& vi : out) {
    for(auto vj : vi)
      std::cout << vj << " ";
    std::cout << "\n";
  }


  auto sum = make_lazy([](auto& out, auto x, auto y, auto z) { out += x + y - std::sin(z); });
  double rsum = 0;
  sum(rsum, x, y, z);
  std::cout << "rsum: " << rsum << "\n";


  /* with points
   *
   * points are not containers (no size) so we need to define a few methods
   * alternatively, we could have typedefed std::array<double, 3<, but that's not... the point :)
   */

  struct point {
    double x,y,z;
    point operator+(point const& other) const {
      return {x + other.x, y + other.y, z + other.z};
    }
    point operator+(double v) const {
      return {x + v, y + v, z + v};
    }
    point operator-(double v) const {
      return {x - v, y - v, z - v};
    }
  };

  std::array<point, 1> px = {1,2,3}, py = {4,5,6};
  float pz=12.;
  std::array<point, 1> pout;
  op(pout, px, py, pz);

  for(auto p : pout)
    std::cout << "(" << p.x << ", " << p.y << ", " << p.z << ") ";
  std::cout << "\n";
  return 0;
}


