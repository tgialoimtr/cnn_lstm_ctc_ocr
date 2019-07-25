#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "column.h"

namespace py = pybind11;

PYBIND11_MODULE(location_nn, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: location_nn
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

    py::class_<TemplateFilterer>(m, "TemplateFilterer")
            .def(py::init<>())
            .def("add", &TemplateFilterer::add)
            .def("search", &TemplateFilterer::search);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}