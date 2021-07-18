#pragma once

#include <pybind11/numpy.h>

pybind11::array_t<int> cluster_assign(const pybind11::array_t<double>& points,
                                   const pybind11::array_t<double>& weights_,
                                   const pybind11::array_t<double>& centers);