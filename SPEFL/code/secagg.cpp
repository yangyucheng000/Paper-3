// cppimport
#include "../lib/SpeflMath.h"

PYBIND11_MODULE(secagg, m) {
    py::bind_vector< std::vector<Int> >(m, "VectorInt", py::buffer_protocol());
    py::bind_vector< std::vector<uInt> >(m, "VectorUInt", py::buffer_protocol());
    py::bind_vector< std::vector<float> >(m, "Vectorfloat", py::buffer_protocol());
    m.def("vector_add",&vector_add);
    m.def("vector_sub",&vector_sub);
    m.def("get_garble",&get_garble);
    m.def("vector_mul",&vector_mul);
    m.def("vecmat_mul",&vecmat_mul);
    m.def("vecnum_sub",&vecnum_sub);
    m.def("vecflo_mul",&vecflo_mul);
}
/*
<%
setup_pybind11(cfg)
cfg['sources'] = ['../lib/SpeflMath.cpp']
cfg['dependencies'] = ['../lib/SpeflMath.h','../lib/SpeflCommon.h']
%>
*/