template void Argmax
<float, float>(
unsigned option,
func_t<float, float> p_func,
const VectorGPU<float>& vec,
unsigned& result,
unsigned int grid_size,
unsigned int block_size);
