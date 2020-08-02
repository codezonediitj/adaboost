template void Argmax
<float, float>(
unsigned option,
const VectorGPU<float>& vec,
unsigned& result,
unsigned int grid_size,
unsigned int block_size,
float val);
