import pycuda.driver as cuda
import pycuda.autoinit, pycuda.compiler
import numpy as np

a = np.random.rand(4,4).astype(np.float32)
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

#compute kernel
mod = pycuda.compiler. SourceModule("""
    __glovabl__ void twice(float * a)
    {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
    }
    """)

func = mod.get_function("twice")
func(a_gpu, block=(4,4,1))

a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print(a_doubled)
print(a)
