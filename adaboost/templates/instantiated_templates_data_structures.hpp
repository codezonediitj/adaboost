#ifndef ADABOOST_CORE_INSTANTIATED_TEMPLATES_DATA_STRUCTURES_HPP
#define ADABOOST_CORE_INSTANTIATED_TEMPLATES_DATA_STRUCTURES_HPP

template class Vector<bool>;
template class Vector<short>;
template class Vector<unsigned short>;
template class Vector<int>;
template class Vector<unsigned int>;
template class Vector<long>;
template class Vector<unsigned long>;
template class Vector<long long>;
template class Vector<unsigned long long>;
template class Vector<float>;
template class Vector<double>;
template class Vector<long double>;
template class Matrix<bool>;
template class Matrix<short>;
template class Matrix<unsigned short>;
template class Matrix<int>;
template class Matrix<unsigned int>;
template class Matrix<long>;
template class Matrix<unsigned long>;
template class Matrix<long long>;
template class Matrix<unsigned long long>;
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<long double>;
template void product<bool>
(const Vector<bool>&, const Vector<bool>&, bool&);
template void product<short>
(const Vector<short>&, const Vector<short>&, short&);
template void product<unsigned short>
(const Vector<unsigned short>&, const Vector<unsigned short>&, unsigned short&);
template void product<int>
(const Vector<int>&, const Vector<int>&, int&);
template void product<unsigned int>
(const Vector<unsigned int>&, const Vector<unsigned int>&, unsigned int&);
template void product<long>
(const Vector<long>&, const Vector<long>&, long&);
template void product<unsigned long>
(const Vector<unsigned long>&, const Vector<unsigned long>&, unsigned long&);
template void product<long long>
(const Vector<long long>&, const Vector<long long>&, long long&);
template void product<unsigned long long>
(const Vector<unsigned long long>&, const Vector<unsigned long long>&, unsigned long long&);
template void product<float>
(const Vector<float>&, const Vector<float>&, float&);
template void product<double>
(const Vector<double>&, const Vector<double>&, double&);
template void product<long double>
(const Vector<long double>&, const Vector<long double>&, long double&);
template void multiply<bool>
(const Matrix<bool>&, const Matrix<bool>&, Matrix<bool>&);
template void multiply<short>
(const Matrix<short>&, const Matrix<short>&, Matrix<short>&);
template void multiply<unsigned short>
(const Matrix<unsigned short>&, const Matrix<unsigned short>&, Matrix<unsigned short>&);
template void multiply<int>
(const Matrix<int>&, const Matrix<int>&, Matrix<int>&);
template void multiply<unsigned int>
(const Matrix<unsigned int>&, const Matrix<unsigned int>&, Matrix<unsigned int>&);
template void multiply<long>
(const Matrix<long>&, const Matrix<long>&, Matrix<long>&);
template void multiply<unsigned long>
(const Matrix<unsigned long>&, const Matrix<unsigned long>&, Matrix<unsigned long>&);
template void multiply<long long>
(const Matrix<long long>&, const Matrix<long long>&, Matrix<long long>&);
template void multiply<unsigned long long>
(const Matrix<unsigned long long>&, const Matrix<unsigned long long>&, Matrix<unsigned long long>&);
template void multiply<float>
(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void multiply<double>
(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
template void multiply<long double>
(const Matrix<long double>&, const Matrix<long double>&, Matrix<long double>&);
template void multiply<bool, bool>
(const Vector<bool>&, const Matrix<bool>&, Vector<bool>&);
template void multiply<bool, short>
(const Vector<bool>&, const Matrix<short>&, Vector<bool>&);
template void multiply<bool, unsigned short>
(const Vector<bool>&, const Matrix<unsigned short>&, Vector<bool>&);
template void multiply<bool, int>
(const Vector<bool>&, const Matrix<int>&, Vector<bool>&);
template void multiply<bool, unsigned int>
(const Vector<bool>&, const Matrix<unsigned int>&, Vector<bool>&);
template void multiply<bool, long>
(const Vector<bool>&, const Matrix<long>&, Vector<bool>&);
template void multiply<bool, unsigned long>
(const Vector<bool>&, const Matrix<unsigned long>&, Vector<bool>&);
template void multiply<bool, long long>
(const Vector<bool>&, const Matrix<long long>&, Vector<bool>&);
template void multiply<bool, unsigned long long>
(const Vector<bool>&, const Matrix<unsigned long long>&, Vector<bool>&);
template void multiply<bool, float>
(const Vector<bool>&, const Matrix<float>&, Vector<bool>&);
template void multiply<bool, double>
(const Vector<bool>&, const Matrix<double>&, Vector<bool>&);
template void multiply<bool, long double>
(const Vector<bool>&, const Matrix<long double>&, Vector<bool>&);
template void multiply<short, bool>
(const Vector<short>&, const Matrix<bool>&, Vector<short>&);
template void multiply<short, short>
(const Vector<short>&, const Matrix<short>&, Vector<short>&);
template void multiply<short, unsigned short>
(const Vector<short>&, const Matrix<unsigned short>&, Vector<short>&);
template void multiply<short, int>
(const Vector<short>&, const Matrix<int>&, Vector<short>&);
template void multiply<short, unsigned int>
(const Vector<short>&, const Matrix<unsigned int>&, Vector<short>&);
template void multiply<short, long>
(const Vector<short>&, const Matrix<long>&, Vector<short>&);
template void multiply<short, unsigned long>
(const Vector<short>&, const Matrix<unsigned long>&, Vector<short>&);
template void multiply<short, long long>
(const Vector<short>&, const Matrix<long long>&, Vector<short>&);
template void multiply<short, unsigned long long>
(const Vector<short>&, const Matrix<unsigned long long>&, Vector<short>&);
template void multiply<short, float>
(const Vector<short>&, const Matrix<float>&, Vector<short>&);
template void multiply<short, double>
(const Vector<short>&, const Matrix<double>&, Vector<short>&);
template void multiply<short, long double>
(const Vector<short>&, const Matrix<long double>&, Vector<short>&);
template void multiply<unsigned short, bool>
(const Vector<unsigned short>&, const Matrix<bool>&, Vector<unsigned short>&);
template void multiply<unsigned short, short>
(const Vector<unsigned short>&, const Matrix<short>&, Vector<unsigned short>&);
template void multiply<unsigned short, unsigned short>
(const Vector<unsigned short>&, const Matrix<unsigned short>&, Vector<unsigned short>&);
template void multiply<unsigned short, int>
(const Vector<unsigned short>&, const Matrix<int>&, Vector<unsigned short>&);
template void multiply<unsigned short, unsigned int>
(const Vector<unsigned short>&, const Matrix<unsigned int>&, Vector<unsigned short>&);
template void multiply<unsigned short, long>
(const Vector<unsigned short>&, const Matrix<long>&, Vector<unsigned short>&);
template void multiply<unsigned short, unsigned long>
(const Vector<unsigned short>&, const Matrix<unsigned long>&, Vector<unsigned short>&);
template void multiply<unsigned short, long long>
(const Vector<unsigned short>&, const Matrix<long long>&, Vector<unsigned short>&);
template void multiply<unsigned short, unsigned long long>
(const Vector<unsigned short>&, const Matrix<unsigned long long>&, Vector<unsigned short>&);
template void multiply<unsigned short, float>
(const Vector<unsigned short>&, const Matrix<float>&, Vector<unsigned short>&);
template void multiply<unsigned short, double>
(const Vector<unsigned short>&, const Matrix<double>&, Vector<unsigned short>&);
template void multiply<unsigned short, long double>
(const Vector<unsigned short>&, const Matrix<long double>&, Vector<unsigned short>&);
template void multiply<int, bool>
(const Vector<int>&, const Matrix<bool>&, Vector<int>&);
template void multiply<int, short>
(const Vector<int>&, const Matrix<short>&, Vector<int>&);
template void multiply<int, unsigned short>
(const Vector<int>&, const Matrix<unsigned short>&, Vector<int>&);
template void multiply<int, int>
(const Vector<int>&, const Matrix<int>&, Vector<int>&);
template void multiply<int, unsigned int>
(const Vector<int>&, const Matrix<unsigned int>&, Vector<int>&);
template void multiply<int, long>
(const Vector<int>&, const Matrix<long>&, Vector<int>&);
template void multiply<int, unsigned long>
(const Vector<int>&, const Matrix<unsigned long>&, Vector<int>&);
template void multiply<int, long long>
(const Vector<int>&, const Matrix<long long>&, Vector<int>&);
template void multiply<int, unsigned long long>
(const Vector<int>&, const Matrix<unsigned long long>&, Vector<int>&);
template void multiply<int, float>
(const Vector<int>&, const Matrix<float>&, Vector<int>&);
template void multiply<int, double>
(const Vector<int>&, const Matrix<double>&, Vector<int>&);
template void multiply<int, long double>
(const Vector<int>&, const Matrix<long double>&, Vector<int>&);
template void multiply<unsigned int, bool>
(const Vector<unsigned int>&, const Matrix<bool>&, Vector<unsigned int>&);
template void multiply<unsigned int, short>
(const Vector<unsigned int>&, const Matrix<short>&, Vector<unsigned int>&);
template void multiply<unsigned int, unsigned short>
(const Vector<unsigned int>&, const Matrix<unsigned short>&, Vector<unsigned int>&);
template void multiply<unsigned int, int>
(const Vector<unsigned int>&, const Matrix<int>&, Vector<unsigned int>&);
template void multiply<unsigned int, unsigned int>
(const Vector<unsigned int>&, const Matrix<unsigned int>&, Vector<unsigned int>&);
template void multiply<unsigned int, long>
(const Vector<unsigned int>&, const Matrix<long>&, Vector<unsigned int>&);
template void multiply<unsigned int, unsigned long>
(const Vector<unsigned int>&, const Matrix<unsigned long>&, Vector<unsigned int>&);
template void multiply<unsigned int, long long>
(const Vector<unsigned int>&, const Matrix<long long>&, Vector<unsigned int>&);
template void multiply<unsigned int, unsigned long long>
(const Vector<unsigned int>&, const Matrix<unsigned long long>&, Vector<unsigned int>&);
template void multiply<unsigned int, float>
(const Vector<unsigned int>&, const Matrix<float>&, Vector<unsigned int>&);
template void multiply<unsigned int, double>
(const Vector<unsigned int>&, const Matrix<double>&, Vector<unsigned int>&);
template void multiply<unsigned int, long double>
(const Vector<unsigned int>&, const Matrix<long double>&, Vector<unsigned int>&);
template void multiply<long, bool>
(const Vector<long>&, const Matrix<bool>&, Vector<long>&);
template void multiply<long, short>
(const Vector<long>&, const Matrix<short>&, Vector<long>&);
template void multiply<long, unsigned short>
(const Vector<long>&, const Matrix<unsigned short>&, Vector<long>&);
template void multiply<long, int>
(const Vector<long>&, const Matrix<int>&, Vector<long>&);
template void multiply<long, unsigned int>
(const Vector<long>&, const Matrix<unsigned int>&, Vector<long>&);
template void multiply<long, long>
(const Vector<long>&, const Matrix<long>&, Vector<long>&);
template void multiply<long, unsigned long>
(const Vector<long>&, const Matrix<unsigned long>&, Vector<long>&);
template void multiply<long, long long>
(const Vector<long>&, const Matrix<long long>&, Vector<long>&);
template void multiply<long, unsigned long long>
(const Vector<long>&, const Matrix<unsigned long long>&, Vector<long>&);
template void multiply<long, float>
(const Vector<long>&, const Matrix<float>&, Vector<long>&);
template void multiply<long, double>
(const Vector<long>&, const Matrix<double>&, Vector<long>&);
template void multiply<long, long double>
(const Vector<long>&, const Matrix<long double>&, Vector<long>&);
template void multiply<unsigned long, bool>
(const Vector<unsigned long>&, const Matrix<bool>&, Vector<unsigned long>&);
template void multiply<unsigned long, short>
(const Vector<unsigned long>&, const Matrix<short>&, Vector<unsigned long>&);
template void multiply<unsigned long, unsigned short>
(const Vector<unsigned long>&, const Matrix<unsigned short>&, Vector<unsigned long>&);
template void multiply<unsigned long, int>
(const Vector<unsigned long>&, const Matrix<int>&, Vector<unsigned long>&);
template void multiply<unsigned long, unsigned int>
(const Vector<unsigned long>&, const Matrix<unsigned int>&, Vector<unsigned long>&);
template void multiply<unsigned long, long>
(const Vector<unsigned long>&, const Matrix<long>&, Vector<unsigned long>&);
template void multiply<unsigned long, unsigned long>
(const Vector<unsigned long>&, const Matrix<unsigned long>&, Vector<unsigned long>&);
template void multiply<unsigned long, long long>
(const Vector<unsigned long>&, const Matrix<long long>&, Vector<unsigned long>&);
template void multiply<unsigned long, unsigned long long>
(const Vector<unsigned long>&, const Matrix<unsigned long long>&, Vector<unsigned long>&);
template void multiply<unsigned long, float>
(const Vector<unsigned long>&, const Matrix<float>&, Vector<unsigned long>&);
template void multiply<unsigned long, double>
(const Vector<unsigned long>&, const Matrix<double>&, Vector<unsigned long>&);
template void multiply<unsigned long, long double>
(const Vector<unsigned long>&, const Matrix<long double>&, Vector<unsigned long>&);
template void multiply<long long, bool>
(const Vector<long long>&, const Matrix<bool>&, Vector<long long>&);
template void multiply<long long, short>
(const Vector<long long>&, const Matrix<short>&, Vector<long long>&);
template void multiply<long long, unsigned short>
(const Vector<long long>&, const Matrix<unsigned short>&, Vector<long long>&);
template void multiply<long long, int>
(const Vector<long long>&, const Matrix<int>&, Vector<long long>&);
template void multiply<long long, unsigned int>
(const Vector<long long>&, const Matrix<unsigned int>&, Vector<long long>&);
template void multiply<long long, long>
(const Vector<long long>&, const Matrix<long>&, Vector<long long>&);
template void multiply<long long, unsigned long>
(const Vector<long long>&, const Matrix<unsigned long>&, Vector<long long>&);
template void multiply<long long, long long>
(const Vector<long long>&, const Matrix<long long>&, Vector<long long>&);
template void multiply<long long, unsigned long long>
(const Vector<long long>&, const Matrix<unsigned long long>&, Vector<long long>&);
template void multiply<long long, float>
(const Vector<long long>&, const Matrix<float>&, Vector<long long>&);
template void multiply<long long, double>
(const Vector<long long>&, const Matrix<double>&, Vector<long long>&);
template void multiply<long long, long double>
(const Vector<long long>&, const Matrix<long double>&, Vector<long long>&);
template void multiply<unsigned long long, bool>
(const Vector<unsigned long long>&, const Matrix<bool>&, Vector<unsigned long long>&);
template void multiply<unsigned long long, short>
(const Vector<unsigned long long>&, const Matrix<short>&, Vector<unsigned long long>&);
template void multiply<unsigned long long, unsigned short>
(const Vector<unsigned long long>&, const Matrix<unsigned short>&, Vector<unsigned long long>&);
template void multiply<unsigned long long, int>
(const Vector<unsigned long long>&, const Matrix<int>&, Vector<unsigned long long>&);
template void multiply<unsigned long long, unsigned int>
(const Vector<unsigned long long>&, const Matrix<unsigned int>&, Vector<unsigned long long>&);
template void multiply<unsigned long long, long>
(const Vector<unsigned long long>&, const Matrix<long>&, Vector<unsigned long long>&);
template void multiply<unsigned long long, unsigned long>
(const Vector<unsigned long long>&, const Matrix<unsigned long>&, Vector<unsigned long long>&);
template void multiply<unsigned long long, long long>
(const Vector<unsigned long long>&, const Matrix<long long>&, Vector<unsigned long long>&);
template void multiply<unsigned long long, unsigned long long>
(const Vector<unsigned long long>&, const Matrix<unsigned long long>&, Vector<unsigned long long>&);
template void multiply<unsigned long long, float>
(const Vector<unsigned long long>&, const Matrix<float>&, Vector<unsigned long long>&);
template void multiply<unsigned long long, double>
(const Vector<unsigned long long>&, const Matrix<double>&, Vector<unsigned long long>&);
template void multiply<unsigned long long, long double>
(const Vector<unsigned long long>&, const Matrix<long double>&, Vector<unsigned long long>&);
template void multiply<float, bool>
(const Vector<float>&, const Matrix<bool>&, Vector<float>&);
template void multiply<float, short>
(const Vector<float>&, const Matrix<short>&, Vector<float>&);
template void multiply<float, unsigned short>
(const Vector<float>&, const Matrix<unsigned short>&, Vector<float>&);
template void multiply<float, int>
(const Vector<float>&, const Matrix<int>&, Vector<float>&);
template void multiply<float, unsigned int>
(const Vector<float>&, const Matrix<unsigned int>&, Vector<float>&);
template void multiply<float, long>
(const Vector<float>&, const Matrix<long>&, Vector<float>&);
template void multiply<float, unsigned long>
(const Vector<float>&, const Matrix<unsigned long>&, Vector<float>&);
template void multiply<float, long long>
(const Vector<float>&, const Matrix<long long>&, Vector<float>&);
template void multiply<float, unsigned long long>
(const Vector<float>&, const Matrix<unsigned long long>&, Vector<float>&);
template void multiply<float, float>
(const Vector<float>&, const Matrix<float>&, Vector<float>&);
template void multiply<float, double>
(const Vector<float>&, const Matrix<double>&, Vector<float>&);
template void multiply<float, long double>
(const Vector<float>&, const Matrix<long double>&, Vector<float>&);
template void multiply<double, bool>
(const Vector<double>&, const Matrix<bool>&, Vector<double>&);
template void multiply<double, short>
(const Vector<double>&, const Matrix<short>&, Vector<double>&);
template void multiply<double, unsigned short>
(const Vector<double>&, const Matrix<unsigned short>&, Vector<double>&);
template void multiply<double, int>
(const Vector<double>&, const Matrix<int>&, Vector<double>&);
template void multiply<double, unsigned int>
(const Vector<double>&, const Matrix<unsigned int>&, Vector<double>&);
template void multiply<double, long>
(const Vector<double>&, const Matrix<long>&, Vector<double>&);
template void multiply<double, unsigned long>
(const Vector<double>&, const Matrix<unsigned long>&, Vector<double>&);
template void multiply<double, long long>
(const Vector<double>&, const Matrix<long long>&, Vector<double>&);
template void multiply<double, unsigned long long>
(const Vector<double>&, const Matrix<unsigned long long>&, Vector<double>&);
template void multiply<double, float>
(const Vector<double>&, const Matrix<float>&, Vector<double>&);
template void multiply<double, double>
(const Vector<double>&, const Matrix<double>&, Vector<double>&);
template void multiply<double, long double>
(const Vector<double>&, const Matrix<long double>&, Vector<double>&);
template void multiply<long double, bool>
(const Vector<long double>&, const Matrix<bool>&, Vector<long double>&);
template void multiply<long double, short>
(const Vector<long double>&, const Matrix<short>&, Vector<long double>&);
template void multiply<long double, unsigned short>
(const Vector<long double>&, const Matrix<unsigned short>&, Vector<long double>&);
template void multiply<long double, int>
(const Vector<long double>&, const Matrix<int>&, Vector<long double>&);
template void multiply<long double, unsigned int>
(const Vector<long double>&, const Matrix<unsigned int>&, Vector<long double>&);
template void multiply<long double, long>
(const Vector<long double>&, const Matrix<long>&, Vector<long double>&);
template void multiply<long double, unsigned long>
(const Vector<long double>&, const Matrix<unsigned long>&, Vector<long double>&);
template void multiply<long double, long long>
(const Vector<long double>&, const Matrix<long long>&, Vector<long double>&);
template void multiply<long double, unsigned long long>
(const Vector<long double>&, const Matrix<unsigned long long>&, Vector<long double>&);
template void multiply<long double, float>
(const Vector<long double>&, const Matrix<float>&, Vector<long double>&);
template void multiply<long double, double>
(const Vector<long double>&, const Matrix<double>&, Vector<long double>&);
template void multiply<long double, long double>
(const Vector<long double>&, const Matrix<long double>&, Vector<long double>&);

#endif
