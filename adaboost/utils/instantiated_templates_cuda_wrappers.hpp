template void cuda_memcpy<bool>
(bool* ptr_1, bool* ptr_2, unsigned num_bytes, direction d);
template void cuda_memcpy<short>
(short* ptr_1, short* ptr_2, unsigned num_bytes, direction d);
template void cuda_memcpy<unsigned short>
(unsigned short* ptr_1, unsigned short* ptr_2, unsigned num_bytes, direction d);
template void cuda_memcpy<int>
(int* ptr_1, int* ptr_2, unsigned num_bytes, direction d);
template void cuda_memcpy<unsigned int>
(unsigned int* ptr_1, unsigned int* ptr_2, unsigned num_bytes, direction d);
template void cuda_memcpy<long>
(long* ptr_1, long* ptr_2, unsigned num_bytes, direction d);
template void cuda_memcpy<unsigned long>
(unsigned long* ptr_1, unsigned long* ptr_2, unsigned num_bytes, direction d);
template void cuda_memcpy<long long>
(long long* ptr_1, long long* ptr_2, unsigned num_bytes, direction d);
template void cuda_memcpy<unsigned long long>
(unsigned long long* ptr_1, unsigned long long* ptr_2, unsigned num_bytes, direction d);
template void cuda_memcpy<float>
(float* ptr_1, float* ptr_2, unsigned num_bytes, direction d);
template void cuda_memcpy<double>
(double* ptr_1, double* ptr_2, unsigned num_bytes, direction d);
template void cuda_memcpy<long double>
(long double* ptr_1, long double* ptr_2, unsigned num_bytes, direction d);
