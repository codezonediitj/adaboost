template void product_gpu<bool>(
const VectorGPU<bool>&, const VectorGPU<bool>&, bool& result, unsigned);
template void product_gpu<short>(
const VectorGPU<short>&, const VectorGPU<short>&, short& result, unsigned);
template void product_gpu<unsigned short>(
const VectorGPU<unsigned short>&, const VectorGPU<unsigned short>&, unsigned short& result, unsigned);
template void product_gpu<int>(
const VectorGPU<int>&, const VectorGPU<int>&, int& result, unsigned);
template void product_gpu<unsigned int>(
const VectorGPU<unsigned int>&, const VectorGPU<unsigned int>&, unsigned int& result, unsigned);
template void product_gpu<long>(
const VectorGPU<long>&, const VectorGPU<long>&, long& result, unsigned);
template void product_gpu<unsigned long>(
const VectorGPU<unsigned long>&, const VectorGPU<unsigned long>&, unsigned long& result, unsigned);
template void product_gpu<long long>(
const VectorGPU<long long>&, const VectorGPU<long long>&, long long& result, unsigned);
template void product_gpu<unsigned long long>(
const VectorGPU<unsigned long long>&, const VectorGPU<unsigned long long>&, unsigned long long& result, unsigned);
template void product_gpu<float>(
const VectorGPU<float>&, const VectorGPU<float>&, float& result, unsigned);
template void product_gpu<double>(
const VectorGPU<double>&, const VectorGPU<double>&, double& result, unsigned);
template void product_gpu<long double>(
const VectorGPU<long double>&, const VectorGPU<long double>&, long double& result, unsigned);
template void multiply_gpu
(const MatrixGPU<bool>& mat1,
const MatrixGPU<bool>& mat2,
MatrixGPU<bool>& result);
template void multiply_gpu
(const MatrixGPU<short>& mat1,
const MatrixGPU<short>& mat2,
MatrixGPU<short>& result);
template void multiply_gpu
(const MatrixGPU<unsigned short>& mat1,
const MatrixGPU<unsigned short>& mat2,
MatrixGPU<unsigned short>& result);
template void multiply_gpu
(const MatrixGPU<int>& mat1,
const MatrixGPU<int>& mat2,
MatrixGPU<int>& result);
template void multiply_gpu
(const MatrixGPU<unsigned int>& mat1,
const MatrixGPU<unsigned int>& mat2,
MatrixGPU<unsigned int>& result);
template void multiply_gpu
(const MatrixGPU<long>& mat1,
const MatrixGPU<long>& mat2,
MatrixGPU<long>& result);
template void multiply_gpu
(const MatrixGPU<unsigned long>& mat1,
const MatrixGPU<unsigned long>& mat2,
MatrixGPU<unsigned long>& result);
template void multiply_gpu
(const MatrixGPU<long long>& mat1,
const MatrixGPU<long long>& mat2,
MatrixGPU<long long>& result);
template void multiply_gpu
(const MatrixGPU<unsigned long long>& mat1,
const MatrixGPU<unsigned long long>& mat2,
MatrixGPU<unsigned long long>& result);
template void multiply_gpu
(const MatrixGPU<float>& mat1,
const MatrixGPU<float>& mat2,
MatrixGPU<float>& result);
template void multiply_gpu
(const MatrixGPU<double>& mat1,
const MatrixGPU<double>& mat2,
MatrixGPU<double>& result);
template void multiply_gpu
(const MatrixGPU<long double>& mat1,
const MatrixGPU<long double>& mat2,
MatrixGPU<long double>& result);
template void fill<float>(float, adaboost::cuda::core::VectorGPU<float> const&, unsigned int);
template void fill<float>(float, adaboost::cuda::core::MatrixGPU<float> const&, unsigned int, unsigned int);
template void fill<float>(float, adaboost::cuda::core::MatrixGPU<float> const&, unsigned int);