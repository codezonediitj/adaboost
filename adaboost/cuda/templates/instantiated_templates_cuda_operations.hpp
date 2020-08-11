template void product_gpu<bool>(
VectorGPU<bool>&, VectorGPU<bool>&, bool& result, unsigned);
template void product_gpu<short>(
VectorGPU<short>&, VectorGPU<short>&, short& result, unsigned);
template void product_gpu<unsigned short>(
VectorGPU<unsigned short>&, VectorGPU<unsigned short>&, unsigned short& result, unsigned);
template void product_gpu<int>(
VectorGPU<int>&, VectorGPU<int>&, int& result, unsigned);
template void product_gpu<unsigned int>(
VectorGPU<unsigned int>&, VectorGPU<unsigned int>&, unsigned int& result, unsigned);
template void product_gpu<long>(
VectorGPU<long>&, VectorGPU<long>&, long& result, unsigned);
template void product_gpu<unsigned long>(
VectorGPU<unsigned long>&, VectorGPU<unsigned long>&, unsigned long& result, unsigned);
template void product_gpu<long long>(
VectorGPU<long long>&, VectorGPU<long long>&, long long& result, unsigned);
template void product_gpu<unsigned long long>(
VectorGPU<unsigned long long>&, VectorGPU<unsigned long long>&, unsigned long long& result, unsigned);
template void product_gpu<float>(
VectorGPU<float>&, VectorGPU<float>&, float& result, unsigned);
template void product_gpu<double>(
VectorGPU<double>&, VectorGPU<double>&, double& result, unsigned);
template void product_gpu<long double>(
VectorGPU<long double>&, VectorGPU<long double>&, long double& result, unsigned);
template void multiply_gpu
(MatrixGPU<bool>& mat1,
MatrixGPU<bool>& mat2,
MatrixGPU<bool>& result);
template void multiply_gpu
(MatrixGPU<short>& mat1,
MatrixGPU<short>& mat2,
MatrixGPU<short>& result);
template void multiply_gpu
(MatrixGPU<unsigned short>& mat1,
MatrixGPU<unsigned short>& mat2,
MatrixGPU<unsigned short>& result);
template void multiply_gpu
(MatrixGPU<int>& mat1,
MatrixGPU<int>& mat2,
MatrixGPU<int>& result);
template void multiply_gpu
(MatrixGPU<unsigned int>& mat1,
MatrixGPU<unsigned int>& mat2,
MatrixGPU<unsigned int>& result);
template void multiply_gpu
(MatrixGPU<long>& mat1,
MatrixGPU<long>& mat2,
MatrixGPU<long>& result);
template void multiply_gpu
(MatrixGPU<unsigned long>& mat1,
MatrixGPU<unsigned long>& mat2,
MatrixGPU<unsigned long>& result);
template void multiply_gpu
(MatrixGPU<long long>& mat1,
MatrixGPU<long long>& mat2,
MatrixGPU<long long>& result);
template void multiply_gpu
(MatrixGPU<unsigned long long>& mat1,
MatrixGPU<unsigned long long>& mat2,
MatrixGPU<unsigned long long>& result);
template void multiply_gpu
(MatrixGPU<float>& mat1,
MatrixGPU<float>& mat2,
MatrixGPU<float>& result);
template void multiply_gpu
(MatrixGPU<double>& mat1,
MatrixGPU<double>& mat2,
MatrixGPU<double>& result);
template void multiply_gpu
(MatrixGPU<long double>& mat1,
MatrixGPU<long double>& mat2,
MatrixGPU<long double>& result);
template void fill(bool, VectorGPU<bool>&, unsigned int);
template void fill(short, VectorGPU<short>&, unsigned int);
template void fill(unsigned short, VectorGPU<unsigned short>&, unsigned int);
template void fill(int, VectorGPU<int>&, unsigned int);
template void fill(unsigned int, VectorGPU<unsigned int>&, unsigned int);
template void fill(long, VectorGPU<long>&, unsigned int);
template void fill(unsigned long, VectorGPU<unsigned long>&, unsigned int);
template void fill(long long, VectorGPU<long long>&, unsigned int);
template void fill(unsigned long long, VectorGPU<unsigned long long>&, unsigned int);
template void fill(float, VectorGPU<float>&, unsigned int);
template void fill(double, VectorGPU<double>&, unsigned int);
template void fill(long double, VectorGPU<long double>&, unsigned int);
template void fill(bool value, MatrixGPU<bool>&, unsigned block_size_x, unsigned block_size_y);
template void fill(short value, MatrixGPU<short>&, unsigned block_size_x, unsigned block_size_y);
template void fill(unsigned short value, MatrixGPU<unsigned short>&, unsigned block_size_x, unsigned block_size_y);
template void fill(int value, MatrixGPU<int>&, unsigned block_size_x, unsigned block_size_y);
template void fill(unsigned int value, MatrixGPU<unsigned int>&, unsigned block_size_x, unsigned block_size_y);
template void fill(long value, MatrixGPU<long>&, unsigned block_size_x, unsigned block_size_y);
template void fill(unsigned long value, MatrixGPU<unsigned long>&, unsigned block_size_x, unsigned block_size_y);
template void fill(long long value, MatrixGPU<long long>&, unsigned block_size_x, unsigned block_size_y);
template void fill(unsigned long long value, MatrixGPU<unsigned long long>&, unsigned block_size_x, unsigned block_size_y);
template void fill(float value, MatrixGPU<float>&, unsigned block_size_x, unsigned block_size_y);
template void fill(double value, MatrixGPU<double>&, unsigned block_size_x, unsigned block_size_y);
template void fill(long double value, MatrixGPU<long double>&, unsigned block_size_x, unsigned block_size_y);
template void fill(bool value, MatrixGPU<bool>&, unsigned num_streams);
template void fill(short value, MatrixGPU<short>&, unsigned num_streams);
template void fill(unsigned short value, MatrixGPU<unsigned short>&, unsigned num_streams);
template void fill(int value, MatrixGPU<int>&, unsigned num_streams);
template void fill(unsigned int value, MatrixGPU<unsigned int>&, unsigned num_streams);
template void fill(long value, MatrixGPU<long>&, unsigned num_streams);
template void fill(unsigned long value, MatrixGPU<unsigned long>&, unsigned num_streams);
template void fill(long long value, MatrixGPU<long long>&, unsigned num_streams);
template void fill(unsigned long long value, MatrixGPU<unsigned long long>&, unsigned num_streams);
template void fill(float value, MatrixGPU<float>&, unsigned num_streams);
template void fill(double value, MatrixGPU<double>&, unsigned num_streams);
template void fill(long double value, MatrixGPU<long double>&, unsigned num_streams);
template void Argmax
<float, float>(
unsigned option,
const VectorGPU<float>& vec,
unsigned& result,
unsigned int grid_size,
unsigned int block_size,
float* val);
