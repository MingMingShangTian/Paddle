// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <stdint.h>
#include <limits>
#if !defined(_WIN32)
#define PADDLE_ALIGN(x) __attribute__((aligned(x)))
#else
#define PADDLE_ALIGN(x) __declspec(align(x))
#endif

#ifdef PADDLE_WITH_CUDA
#include <cuComplex.h>
#include <thrust/complex.h>
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
#include <hip/hip_complex.h>
#include <thrust/complex.h>  // NOLINT
#endif

#include <cstring>

#include "paddle/fluid/platform/hostdevice.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace Eigen {
template <typename T>
struct NumTraits;
}  // namespace Eigen

namespace paddle {
namespace platform {

template <typename T>
struct PADDLE_ALIGN(sizeof(T) * 2) complex {
 public:
  T real;
  T imag;

  complex() = default;
  complex(const complex& o) = default;
  complex& operator=(const complex& o) = default;
  complex(complex&& o) = default;
  complex& operator=(complex&& o) = default;
  ~complex() = default;

  HOSTDEVICE complex(T real, T imag) : real(real), imag(imag) {}
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

  HOSTDEVICE inline explicit complex(const thrust::complex<T>& c) {
    real = c.real();
    imag = c.imag();
  }

  HOSTDEVICE inline explicit operator thrust::complex<T>() const {
    return thrust::complex<T>(real, imag);
  }

#ifdef PADDLE_WITH_HIP
  HOSTDEVICE inline explicit operator hipFloatComplex() const {
    return make_hipFloatComplex(real, imag);
  }
#else
  HOSTDEVICE inline explicit operator cuFloatComplex() const {
    return make_cuFloatComplex(real, imag);
  }
#endif
#endif

  HOSTDEVICE complex(const float& val) : real(val), imag(0) {}
  HOSTDEVICE complex(const double& val) : real(static_cast<T>(val)), imag(0) {}
  HOSTDEVICE complex(const int& val) : real(static_cast<T>(val)), imag(0) {}
  HOSTDEVICE complex(const int64_t& val) : real(static_cast<T>(val)), imag(0) {}
  HOSTDEVICE complex(const paddle::platform::float16& val)
      : real(static_cast<T>(val)), imag(0) {}

  HOSTDEVICE complex(const paddle::platform::bfloat16& val)
      : real(static_cast<T>(val)), imag(0) {}

  template <typename U = T>
  HOSTDEVICE explicit complex(
      const typename std::enable_if<std::is_same<U, float>::value,
                                    complex<double>>::type& other)
      : real(other.real), imag(other.imag) {}

  template <typename U = T>
  HOSTDEVICE explicit complex(
      const typename std::enable_if<std::is_same<U, double>::value,
                                    complex<float>>::type& other)
      : real(other.real), imag(other.imag) {}

  /*
HOSTDEVICE complex(const complex<double>& val)
  : real(static_cast<T>(val.real)),
    imag(static_cast<T>(val.imag)) {}
    */

  // HOSTDEVICE inline explicit operator std::complex<T>() {
  HOSTDEVICE inline operator std::complex<T>() {
    return std::complex<T>(real, imag);
  }

  /*
  HOSTDEVICE inline explicit complex<T>(const T& val)
      : real(complex(static_cast<T>(val)).real) {}
      */

  HOSTDEVICE inline complex<T>(const std::complex<T> val)
      : real(val.real()), imag(val.imag()) {}

  HOSTDEVICE inline complex<T>& operator=(bool b) {
    real = b ? 1 : 0;
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex& operator=(int8_t val) {
    real = static_cast<T>(val);
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex& operator=(uint8_t val) {
    real = static_cast<T>(val);
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex& operator=(int16_t val) {
    real = static_cast<T>(val);
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex& operator=(uint16_t val) {
    real = static_cast<T>(val);
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex& operator=(int32_t val) {
    real = static_cast<T>(val);
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex& operator=(uint32_t val) {
    real = static_cast<T>(val);
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex& operator=(int64_t val) {
    real = static_cast<T>(val);
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex& operator=(uint64_t val) {
    real = static_cast<T>(val);
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex& operator=(float val) {
    real = val;
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline complex& operator=(double val) {
    real = static_cast<T>(val);
    imag = 0;
    return *this;
  }

  HOSTDEVICE inline operator float() const {
    return static_cast<float>(this->real);
  }

  HOSTDEVICE inline explicit operator bool() const {
    return static_cast<bool>(this->real) || static_cast<bool>(this->imag);
  }

  HOSTDEVICE inline explicit operator int8_t() const {
    return static_cast<int8_t>(this->real);
  }

  HOSTDEVICE inline explicit operator uint8_t() const {
    return static_cast<uint8_t>(this->real);
  }

  HOSTDEVICE inline explicit operator int16_t() const {
    return static_cast<int16_t>(this->real);
  }

  HOSTDEVICE inline explicit operator uint16_t() const {
    return static_cast<uint16_t>(this->real);
  }

  HOSTDEVICE inline explicit operator int32_t() const {
    return static_cast<int32_t>(this->real);
  }

  HOSTDEVICE inline explicit operator uint32_t() const {
    return static_cast<uint32_t>(this->real);
  }

  HOSTDEVICE inline explicit operator int64_t() const {
    return static_cast<int64_t>(this->real);
  }

  HOSTDEVICE inline explicit operator uint64_t() const {
    return static_cast<uint64_t>(this->real);
  }

  HOSTDEVICE inline explicit operator double() const {
    return static_cast<double>(this->real);
  }

  HOSTDEVICE inline operator complex<double>() const {
    return complex<double>(static_cast<double>(this->real),
                           static_cast<double>(this->imag));
  }
};

template <typename T>
HOSTDEVICE inline complex<T> operator+(const complex<T>& a,
                                       const complex<T>& b) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex<T>(thrust::complex<T>(a.real, a.imag) +
                    thrust::complex<T>(b.real, b.imag));
#else
  return complex<T>(a.real + b.real, a.imag + b.imag);
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> operator-(const complex<T>& a,
                                       const complex<T>& b) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex<T>(thrust::complex<T>(a.real, a.imag) -
                    thrust::complex<T>(b.real, b.imag));
#else
  return complex<T>(a.real - b.real, a.imag - b.imag);
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> operator*(const complex<T>& a,
                                       const complex<T>& b) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex<T>(thrust::complex<T>(a.real, a.imag) *
                    thrust::complex<T>(b.real, b.imag));
#else
  return complex<T>(a.real * b.real - a.imag * b.imag,
                    a.imag * b.real + b.imag * a.real);
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> operator/(const complex<T>& a,
                                       const complex<T>& b) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex<T>(thrust::complex<T>(a.real, a.imag) /
                    thrust::complex<T>(b.real, b.imag));
#else
  T denominator = b.real * b.real + b.imag * b.imag;
  return complex<T>((a.real * b.real + a.imag * b.imag) / denominator,
                    (a.imag * b.real - a.real * b.imag) / denominator);
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> operator-(const complex<T>& a) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex<T>(-thrust::complex<T>(a.real, a.imag));
#else
  complex<T> res;
  res.real = -a.real;
  res.imag = -a.imag;
  return res;
#endif
}

template <typename T>
HOSTDEVICE inline complex<T>& operator+=(complex<T>& a,  // NOLINT
                                         const complex<T>& b) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  a = complex<T>(thrust::complex<T>(a.real, a.imag) +=
                 thrust::complex<T>(b.real, b.imag));
  return a;
#else
  a.real += b.real;
  a.imag += b.imag;
  return a;
#endif
}

template <typename T>
HOSTDEVICE inline complex<T>& operator-=(complex<T>& a,  // NOLINT
                                         const complex<T>& b) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  a = complex<T>(thrust::complex<T>(a.real, a.imag) -=
                 thrust::complex<T>(b.real, b.imag));
  return a;
#else
  a.real -= b.real;
  a.imag -= b.imag;
  return a;
#endif
}

template <typename T>
HOSTDEVICE inline complex<T>& operator*=(complex<T>& a,  // NOLINT
                                         const complex<T>& b) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  a = complex<T>(thrust::complex<T>(a.real, a.imag) *=
                 thrust::complex<T>(b.real, b.imag));
  return a;
#else
  a.real = a.real * b.real - a.imag * b.imag;
  a.imag = a.imag * b.real + b.imag * a.real;
  return a;
#endif
}

template <typename T>
HOSTDEVICE inline complex<T>& operator/=(complex<T>& a,  // NOLINT
                                         const complex<T>& b) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  a = complex<T>(thrust::complex<T>(a.real, a.imag) /=
                 thrust::complex<T>(b.real, b.imag));
  return a;
#else
  T denominator = b.real * b.real + b.imag * b.imag;
  a.real = (a.real * b.real + a.imag * b.imag) / denominator;
  a.imag = (a.imag * b.real - a.real * b.imag) / denominator;
  return a;
#endif
}

template <typename T>
HOSTDEVICE inline complex<T> raw_uint16_to_complex(uint16_t a) {
  complex<T> res;
  res.real = a;
  return res;
}

template <typename T>
HOSTDEVICE inline bool operator==(const complex<T>& a, const complex<T>& b) {
  return a.real == b.real && a.imag == b.imag;
}

template <typename T>
HOSTDEVICE inline bool operator!=(const complex<T>& a, const complex<T>& b) {
  return a.real != b.real || a.imag != b.imag;
}

template <typename T>
HOSTDEVICE inline bool operator<(const complex<T>& a, const complex<T>& b) {
  return static_cast<T>(a.real) < static_cast<T>(b.real);
}

template <typename T>
HOSTDEVICE inline bool operator<=(const complex<T>& a, const complex<T>& b) {
  return static_cast<T>(a.real) <= static_cast<T>(b.real);
}

template <typename T>
HOSTDEVICE inline bool operator>(const complex<T>& a, const complex<T>& b) {
  return static_cast<T>(a.real) > static_cast<T>(b.real);
}

template <typename T>
HOSTDEVICE inline bool operator>=(const complex<T>& a, const complex<T>& b) {
  return static_cast<T>(a.real) >= static_cast<T>(b.real);
}

template <typename T>
HOSTDEVICE inline bool(isnan)(const complex<T>& a) {
#if defined(__CUDA_ARCH__)
  // __isnanf not supported on HIP platform
  return __isnanf(a.real) || __isnanf(a.imag);
#else
  // return std::isnan<T>(a.real) || std::isnan(a.imag);
  return std::isnan(a.real) || std::isnan(a.imag);
#endif
}

template <typename T>
HOSTDEVICE inline bool isinf(const complex<T>& a) {
#if defined(__CUDA_ARCH__)
  // __isinff not supported on HIP platform
  return __isinff(a.real) || __isinff(a.imag);
#else
  return std::isinf(a.real) || std::isinf(a.imag);
#endif
}

template <typename T>
HOSTDEVICE inline bool(isfinite)(const complex<T>& a) {
  return !((isnan)(a)) && !((isinf)(a));
}

template <typename T>
HOSTDEVICE inline float(abs)(const complex<T>& a) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex<T>(thrust::abs(thrust::complex<T>(a.real, a.imag)));
#else
  return std::abs(std::complex<T>(a.real, a.imag));
#endif
}

template <typename T>
HOSTDEVICE inline complex<T>(pow)(const complex<T>& a, const complex<T>& b) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex<T>(thrust::pow(thrust::complex<T>(a.real, a.imag),
                                thrust::complex<T>(b.real, b.imag)));
#else
  return std::pow(std::complex<T>(a), std::complex<T>(b));
#endif
}

template <typename T>
HOSTDEVICE inline complex<T>(sqrt)(const complex<T>& a) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex<T>(thrust::sqrt(thrust::complex<T>(a.real, a.imag)));
#else
  return std::sqrt(std::complex<T>(a));
#endif
}

template <typename T>
HOSTDEVICE inline complex<T>(tanh)(const complex<T>& a) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex<T>(thrust::tanh(thrust::complex<T>(a.real, a.imag)));
#else
  return std::tanh(std::complex<T>(a));
#endif
}

template <typename T>
HOSTDEVICE inline complex<T>(log)(const complex<T>& a) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return complex<T>(thrust::log(thrust::complex<T>(a.real, a.imag)));
#else
  return std::log(std::complex<T>(a));
#endif
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const complex<T>& a) {
  os << "real:" << a.real << " imag:" << a.imag;
  return os;
}

}  // namespace platform
}  // namespace paddle

namespace std {

template <typename T>
struct is_pod<paddle::platform::complex<T>> {
  static const bool value =
      is_trivial<paddle::platform::complex<T>>::value &&
      is_standard_layout<paddle::platform::complex<T>>::value;
};

template <typename T>
struct is_floating_point<paddle::platform::complex<T>>
    : std::integral_constant<
          bool, std::is_same<paddle::platform::complex<T>,
                             typename std::remove_cv<
                                 paddle::platform::complex<T>>::type>::value> {
};
template <typename T>
struct is_signed<paddle::platform::complex<T>> {
  static const bool value = false;
};

template <typename T>
struct is_unsigned<paddle::platform::complex<T>> {
  static const bool value = false;
};

template <typename T>
inline bool isnan(const paddle::platform::complex<T>& a) {
  return paddle::platform::isnan(a);
}

template <typename T>
inline bool isinf(const paddle::platform::complex<T>& a) {
  return paddle::platform::isinf(a);
}

template <typename T>
struct numeric_limits<paddle::platform::complex<T>> {
  static const bool is_specialized = false;
  static const bool is_signed = false;
  static const bool is_integer = false;
  static const bool is_exact = false;
  static const bool has_infinity = false;
  static const bool has_quiet_NaN = false;
  static const bool has_signaling_NaN = false;
  static const float_denorm_style has_denorm = denorm_absent;
  static const bool has_denorm_loss = false;
  static const std::float_round_style round_style = std::round_toward_zero;
  static const bool is_iec559 = false;
  static const bool is_bounded = false;
  static const bool is_modulo = false;
  static const int digits = 0;
  static const int digits10 = 0;
  static const int max_digits10 = 0;
  static const int radix = 0;
  static const int min_exponent = 0;
  static const int min_exponent10 = 0;
  static const int max_exponent = 0;
  static const int max_exponent10 = 0;
  static const bool traps = false;
  static const bool tinyness_before = false;

  static paddle::platform::complex<T>(min)() {
    return paddle::platform::complex<T>(0.0, 0.0);
  }

  static paddle::platform::complex<T> lowest() {
    return paddle::platform::complex<T>(0.0, 0.0);
  }

  static paddle::platform::complex<T>(max)() {
    return paddle::platform::complex<T>(0.0, 0.0);
  }

  static paddle::platform::complex<T> epsilon() {
    return paddle::platform::complex<T>(0.0, 0.0);
  }

  static paddle::platform::complex<T> round_error() {
    return paddle::platform::complex<T>(0.0, 0.0);
  }

  static paddle::platform::complex<T> infinity() {
    return paddle::platform::complex<T>(0.0, 0.0);
  }

  static paddle::platform::complex<T> quiet_NaN() {
    return paddle::platform::complex<T>(0.0, 0.0);
  }

  static paddle::platform::complex<T> signaling_NaN() {
    return paddle::platform::complex<T>(0.0, 0.0);
  }

  static paddle::platform::complex<T> denorm_min() {
    return paddle::platform::complex<T>(0.0, 0.0);
  }
};

}  // namespace std
namespace Eigen {}  // namespace Eigen
