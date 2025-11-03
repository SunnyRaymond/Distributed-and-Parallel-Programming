NVCC		= nvcc
CC		= g++
CU_FLAGS	= -O2 -g --ptxas-options=-v

CU_SOURCES	= hello_cuda.cu

CU_OBJECTS	= $(CU_SOURCES:%.cu=%.o)
CU_PTX		= $(CU_SOURCES:%.cu=%.ptx)

%.o:		%.cu
		$(NVCC) $(CU_FLAGS) -c $< -o $@

%.ptx:		%.cu
		$(NVCC) $(CU_FLAGS) --ptx $< -o $@

hello_cuda:	$(CU_OBJECTS)
		$(NVCC) $^ -o $@

ptx:		$(CU_PTX) 

clean:
		rm -f *.o hello_cuda
