#pragma once

#include <pybind11/numpy.h>

pybind11::array_t<double> BLK17_sensitivity(const pybind11::array_t<double>& points,
                                   const pybind11::array_t<double>& weights_,
                                   const pybind11::array_t<double>& centers,
                                   const double alpha);

pybind11::array_t<double> FL11_sensitivity(const pybind11::array_t<double>& points,
                                   const pybind11::array_t<double>& weights_,
                                   const pybind11::array_t<double>& centers,
                                   const double alpha);