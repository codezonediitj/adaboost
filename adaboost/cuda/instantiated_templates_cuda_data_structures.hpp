template class VectorGPU<bool>;
template class VectorGPU<short>;
template class VectorGPU<unsigned short>;
template class VectorGPU<int>;
template class VectorGPU<unsigned int>;
template class VectorGPU<long>;
template class VectorGPU<unsigned long>;
template class VectorGPU<long long>;
template class VectorGPU<unsigned long long>;
template class VectorGPU<float>;
template class VectorGPU<double>;
template class VectorGPU<long double>;
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
