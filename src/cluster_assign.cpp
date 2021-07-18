#include "cluster_assign.h"

#include <algorithm>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <cmath>

namespace py = pybind11;

namespace {

const char* cluster_assign_doc = R"(cluster_assign)";

} // namespace

py::array_t<int> cluster_assign(const py::array_t<double>& points_,
    const py::array_t<double>& weights_,
    const py::array_t<double>& centers_) {
    if (points_.shape(1) != centers_.shape(1)) {
        throw std::runtime_error(
            "Data points and centers should have the save dimension.");
    }
    if (points_.shape(0) != weights_.shape(0)) {
        throw std::runtime_error(
            "The number of points and weights should be the same.");
    }
    auto points = points_.unchecked<2>();
    auto centers = centers_.unchecked<2>();
    auto weights = weights_.unchecked<1>();

    int n = points.shape(0);
    std::vector<double> dists(n);
    py::array_t<int> assign_ = py::array_t<double>(n);
    auto assign = assign_.mutable_unchecked<1>();

    // determine the cluster assignments for points
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < centers.shape(0); j++) {
            double dist = 0;
            for (int k = 0; k < points.shape(1); k++) {
                dist += pow(points(i, k) - centers(j, k), 2);
            }
            if (j == 0 || dist < dists[i]) {
                dists[i] = dist;
                assign(i) = j;
            }
        }
    }

    return assign_;
}

PYBIND11_MODULE(cluster_assign, m) {
    py::options options;
    options.disable_function_signatures();
    m.def("cluster_assign", cluster_assign, cluster_assign_doc);
}
