#ifndef ADABOOST_CUDA_TEMPLATES_INSTANTIATED_TEMPLATES_CUDA_OPERATIONS_HPP
#define ADABOOST_CORE_TEMPLATES_INSTANTIATED_TEMPLATES_CUDA_OPERATIONS_HPP

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
template void Argmax<bool, bool>(unsigned option,const VectorGPU<bool>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,bool* val);
template void Argmax<bool, short>(unsigned option,const VectorGPU<bool>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,short* val);
template void Argmax<bool, unsigned short>(unsigned option,const VectorGPU<bool>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned short* val);
template void Argmax<bool, int>(unsigned option,const VectorGPU<bool>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,int* val);
template void Argmax<bool, unsigned int>(unsigned option,const VectorGPU<bool>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned int* val);
template void Argmax<bool, long>(unsigned option,const VectorGPU<bool>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long* val);
template void Argmax<bool, unsigned long>(unsigned option,const VectorGPU<bool>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long* val);
template void Argmax<bool, long long>(unsigned option,const VectorGPU<bool>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long long* val);
template void Argmax<bool, unsigned long long>(unsigned option,const VectorGPU<bool>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long long* val);
template void Argmax<bool, float>(unsigned option,const VectorGPU<bool>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,float* val);
template void Argmax<bool, double>(unsigned option,const VectorGPU<bool>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,double* val);
template void Argmax<bool, long double>(unsigned option,const VectorGPU<bool>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long double* val);
template void Argmax<short, bool>(unsigned option,const VectorGPU<short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,bool* val);
template void Argmax<short, short>(unsigned option,const VectorGPU<short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,short* val);
template void Argmax<short, unsigned short>(unsigned option,const VectorGPU<short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned short* val);
template void Argmax<short, int>(unsigned option,const VectorGPU<short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,int* val);
template void Argmax<short, unsigned int>(unsigned option,const VectorGPU<short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned int* val);
template void Argmax<short, long>(unsigned option,const VectorGPU<short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long* val);
template void Argmax<short, unsigned long>(unsigned option,const VectorGPU<short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long* val);
template void Argmax<short, long long>(unsigned option,const VectorGPU<short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long long* val);
template void Argmax<short, unsigned long long>(unsigned option,const VectorGPU<short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long long* val);
template void Argmax<short, float>(unsigned option,const VectorGPU<short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,float* val);
template void Argmax<short, double>(unsigned option,const VectorGPU<short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,double* val);
template void Argmax<short, long double>(unsigned option,const VectorGPU<short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long double* val);
template void Argmax<unsigned short, bool>(unsigned option,const VectorGPU<unsigned short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,bool* val);
template void Argmax<unsigned short, short>(unsigned option,const VectorGPU<unsigned short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,short* val);
template void Argmax<unsigned short, unsigned short>(unsigned option,const VectorGPU<unsigned short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned short* val);
template void Argmax<unsigned short, int>(unsigned option,const VectorGPU<unsigned short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,int* val);
template void Argmax<unsigned short, unsigned int>(unsigned option,const VectorGPU<unsigned short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned int* val);
template void Argmax<unsigned short, long>(unsigned option,const VectorGPU<unsigned short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long* val);
template void Argmax<unsigned short, unsigned long>(unsigned option,const VectorGPU<unsigned short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long* val);
template void Argmax<unsigned short, long long>(unsigned option,const VectorGPU<unsigned short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long long* val);
template void Argmax<unsigned short, unsigned long long>(unsigned option,const VectorGPU<unsigned short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long long* val);
template void Argmax<unsigned short, float>(unsigned option,const VectorGPU<unsigned short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,float* val);
template void Argmax<unsigned short, double>(unsigned option,const VectorGPU<unsigned short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,double* val);
template void Argmax<unsigned short, long double>(unsigned option,const VectorGPU<unsigned short>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long double* val);
template void Argmax<int, bool>(unsigned option,const VectorGPU<int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,bool* val);
template void Argmax<int, short>(unsigned option,const VectorGPU<int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,short* val);
template void Argmax<int, unsigned short>(unsigned option,const VectorGPU<int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned short* val);
template void Argmax<int, int>(unsigned option,const VectorGPU<int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,int* val);
template void Argmax<int, unsigned int>(unsigned option,const VectorGPU<int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned int* val);
template void Argmax<int, long>(unsigned option,const VectorGPU<int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long* val);
template void Argmax<int, unsigned long>(unsigned option,const VectorGPU<int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long* val);
template void Argmax<int, long long>(unsigned option,const VectorGPU<int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long long* val);
template void Argmax<int, unsigned long long>(unsigned option,const VectorGPU<int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long long* val);
template void Argmax<int, float>(unsigned option,const VectorGPU<int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,float* val);
template void Argmax<int, double>(unsigned option,const VectorGPU<int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,double* val);
template void Argmax<int, long double>(unsigned option,const VectorGPU<int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long double* val);
template void Argmax<unsigned int, bool>(unsigned option,const VectorGPU<unsigned int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,bool* val);
template void Argmax<unsigned int, short>(unsigned option,const VectorGPU<unsigned int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,short* val);
template void Argmax<unsigned int, unsigned short>(unsigned option,const VectorGPU<unsigned int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned short* val);
template void Argmax<unsigned int, int>(unsigned option,const VectorGPU<unsigned int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,int* val);
template void Argmax<unsigned int, unsigned int>(unsigned option,const VectorGPU<unsigned int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned int* val);
template void Argmax<unsigned int, long>(unsigned option,const VectorGPU<unsigned int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long* val);
template void Argmax<unsigned int, unsigned long>(unsigned option,const VectorGPU<unsigned int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long* val);
template void Argmax<unsigned int, long long>(unsigned option,const VectorGPU<unsigned int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long long* val);
template void Argmax<unsigned int, unsigned long long>(unsigned option,const VectorGPU<unsigned int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long long* val);
template void Argmax<unsigned int, float>(unsigned option,const VectorGPU<unsigned int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,float* val);
template void Argmax<unsigned int, double>(unsigned option,const VectorGPU<unsigned int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,double* val);
template void Argmax<unsigned int, long double>(unsigned option,const VectorGPU<unsigned int>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long double* val);
template void Argmax<long, bool>(unsigned option,const VectorGPU<long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,bool* val);
template void Argmax<long, short>(unsigned option,const VectorGPU<long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,short* val);
template void Argmax<long, unsigned short>(unsigned option,const VectorGPU<long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned short* val);
template void Argmax<long, int>(unsigned option,const VectorGPU<long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,int* val);
template void Argmax<long, unsigned int>(unsigned option,const VectorGPU<long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned int* val);
template void Argmax<long, long>(unsigned option,const VectorGPU<long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long* val);
template void Argmax<long, unsigned long>(unsigned option,const VectorGPU<long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long* val);
template void Argmax<long, long long>(unsigned option,const VectorGPU<long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long long* val);
template void Argmax<long, unsigned long long>(unsigned option,const VectorGPU<long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long long* val);
template void Argmax<long, float>(unsigned option,const VectorGPU<long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,float* val);
template void Argmax<long, double>(unsigned option,const VectorGPU<long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,double* val);
template void Argmax<long, long double>(unsigned option,const VectorGPU<long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long double* val);
template void Argmax<unsigned long, bool>(unsigned option,const VectorGPU<unsigned long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,bool* val);
template void Argmax<unsigned long, short>(unsigned option,const VectorGPU<unsigned long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,short* val);
template void Argmax<unsigned long, unsigned short>(unsigned option,const VectorGPU<unsigned long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned short* val);
template void Argmax<unsigned long, int>(unsigned option,const VectorGPU<unsigned long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,int* val);
template void Argmax<unsigned long, unsigned int>(unsigned option,const VectorGPU<unsigned long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned int* val);
template void Argmax<unsigned long, long>(unsigned option,const VectorGPU<unsigned long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long* val);
template void Argmax<unsigned long, unsigned long>(unsigned option,const VectorGPU<unsigned long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long* val);
template void Argmax<unsigned long, long long>(unsigned option,const VectorGPU<unsigned long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long long* val);
template void Argmax<unsigned long, unsigned long long>(unsigned option,const VectorGPU<unsigned long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long long* val);
template void Argmax<unsigned long, float>(unsigned option,const VectorGPU<unsigned long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,float* val);
template void Argmax<unsigned long, double>(unsigned option,const VectorGPU<unsigned long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,double* val);
template void Argmax<unsigned long, long double>(unsigned option,const VectorGPU<unsigned long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long double* val);
template void Argmax<long long, bool>(unsigned option,const VectorGPU<long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,bool* val);
template void Argmax<long long, short>(unsigned option,const VectorGPU<long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,short* val);
template void Argmax<long long, unsigned short>(unsigned option,const VectorGPU<long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned short* val);
template void Argmax<long long, int>(unsigned option,const VectorGPU<long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,int* val);
template void Argmax<long long, unsigned int>(unsigned option,const VectorGPU<long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned int* val);
template void Argmax<long long, long>(unsigned option,const VectorGPU<long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long* val);
template void Argmax<long long, unsigned long>(unsigned option,const VectorGPU<long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long* val);
template void Argmax<long long, long long>(unsigned option,const VectorGPU<long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long long* val);
template void Argmax<long long, unsigned long long>(unsigned option,const VectorGPU<long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long long* val);
template void Argmax<long long, float>(unsigned option,const VectorGPU<long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,float* val);
template void Argmax<long long, double>(unsigned option,const VectorGPU<long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,double* val);
template void Argmax<long long, long double>(unsigned option,const VectorGPU<long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long double* val);
template void Argmax<unsigned long long, bool>(unsigned option,const VectorGPU<unsigned long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,bool* val);
template void Argmax<unsigned long long, short>(unsigned option,const VectorGPU<unsigned long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,short* val);
template void Argmax<unsigned long long, unsigned short>(unsigned option,const VectorGPU<unsigned long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned short* val);
template void Argmax<unsigned long long, int>(unsigned option,const VectorGPU<unsigned long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,int* val);
template void Argmax<unsigned long long, unsigned int>(unsigned option,const VectorGPU<unsigned long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned int* val);
template void Argmax<unsigned long long, long>(unsigned option,const VectorGPU<unsigned long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long* val);
template void Argmax<unsigned long long, unsigned long>(unsigned option,const VectorGPU<unsigned long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long* val);
template void Argmax<unsigned long long, long long>(unsigned option,const VectorGPU<unsigned long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long long* val);
template void Argmax<unsigned long long, unsigned long long>(unsigned option,const VectorGPU<unsigned long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long long* val);
template void Argmax<unsigned long long, float>(unsigned option,const VectorGPU<unsigned long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,float* val);
template void Argmax<unsigned long long, double>(unsigned option,const VectorGPU<unsigned long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,double* val);
template void Argmax<unsigned long long, long double>(unsigned option,const VectorGPU<unsigned long long>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long double* val);
template void Argmax<float, bool>(unsigned option,const VectorGPU<float>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,bool* val);
template void Argmax<float, short>(unsigned option,const VectorGPU<float>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,short* val);
template void Argmax<float, unsigned short>(unsigned option,const VectorGPU<float>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned short* val);
template void Argmax<float, int>(unsigned option,const VectorGPU<float>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,int* val);
template void Argmax<float, unsigned int>(unsigned option,const VectorGPU<float>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned int* val);
template void Argmax<float, long>(unsigned option,const VectorGPU<float>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long* val);
template void Argmax<float, unsigned long>(unsigned option,const VectorGPU<float>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long* val);
template void Argmax<float, long long>(unsigned option,const VectorGPU<float>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long long* val);
template void Argmax<float, unsigned long long>(unsigned option,const VectorGPU<float>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long long* val);
template void Argmax<float, float>(unsigned option,const VectorGPU<float>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,float* val);
template void Argmax<float, double>(unsigned option,const VectorGPU<float>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,double* val);
template void Argmax<float, long double>(unsigned option,const VectorGPU<float>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long double* val);
template void Argmax<double, bool>(unsigned option,const VectorGPU<double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,bool* val);
template void Argmax<double, short>(unsigned option,const VectorGPU<double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,short* val);
template void Argmax<double, unsigned short>(unsigned option,const VectorGPU<double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned short* val);
template void Argmax<double, int>(unsigned option,const VectorGPU<double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,int* val);
template void Argmax<double, unsigned int>(unsigned option,const VectorGPU<double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned int* val);
template void Argmax<double, long>(unsigned option,const VectorGPU<double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long* val);
template void Argmax<double, unsigned long>(unsigned option,const VectorGPU<double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long* val);
template void Argmax<double, long long>(unsigned option,const VectorGPU<double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long long* val);
template void Argmax<double, unsigned long long>(unsigned option,const VectorGPU<double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long long* val);
template void Argmax<double, float>(unsigned option,const VectorGPU<double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,float* val);
template void Argmax<double, double>(unsigned option,const VectorGPU<double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,double* val);
template void Argmax<double, long double>(unsigned option,const VectorGPU<double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long double* val);
template void Argmax<long double, bool>(unsigned option,const VectorGPU<long double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,bool* val);
template void Argmax<long double, short>(unsigned option,const VectorGPU<long double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,short* val);
template void Argmax<long double, unsigned short>(unsigned option,const VectorGPU<long double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned short* val);
template void Argmax<long double, int>(unsigned option,const VectorGPU<long double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,int* val);
template void Argmax<long double, unsigned int>(unsigned option,const VectorGPU<long double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned int* val);
template void Argmax<long double, long>(unsigned option,const VectorGPU<long double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long* val);
template void Argmax<long double, unsigned long>(unsigned option,const VectorGPU<long double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long* val);
template void Argmax<long double, long long>(unsigned option,const VectorGPU<long double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long long* val);
template void Argmax<long double, unsigned long long>(unsigned option,const VectorGPU<long double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,unsigned long long* val);
template void Argmax<long double, float>(unsigned option,const VectorGPU<long double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,float* val);
template void Argmax<long double, double>(unsigned option,const VectorGPU<long double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,double* val);
template void Argmax<long double, long double>(unsigned option,const VectorGPU<long double>& vec,unsigned& result,unsigned int grid_size,unsigned int block_size,long double* val);

#endif
