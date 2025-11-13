# Compilers we'll use
NVCC = nvcc
CC   = g++

# Change this value to change the GPU compute capability your code will be
#   compiled against. Alternatively, compile with 'make COMPUTE=..." to make a
#   one-time change.
ifndef COMPUTE
COMPUTE = 52
endif

# Flags for each of the compilers
CU_FLAGS	= -O3 -g --ptxas-options=-v -arch compute_$(COMPUTE) -code sm_$(COMPUTE)
CC_FLAGS	= -O3 -m64 -Wall

# Define the library sources we'll need for compilation. If you create any new
#   .cc or .cu files used in the assignment, add them here
CU_SOURCES	= simulate.cu
CC_SOURCES	= file.cc timer.cc

# Create paths to their relevant object files
CU_OBJECTS	= $(CU_SOURCES:%.cu=%.o)
CU_PTX		= $(CU_SOURCES:%.cu=%.ptx)
CC_OBJECTS	= $(CC_SOURCES:%.cc=%.o)

# Arguments to run the file with (as i_max, t_max, block_size)
RUNARGS = 1000000 1000 32



##### COMPILE RULES #####

# Mark certain rules as 'phony', i.e., not tied to actual files
.PHONY: default ptx run runlocal plot all clean
# Define the default rule to compile
default: assign2_1
# Rule to compile all 'major' rules
all: assign2_1 vector-add

# Compile objects; either as CUDA objects (.cu), C++ objects (.cc) or GPU assembly
%.o: %.cu
	$(NVCC) $(CU_FLAGS) -c $< -o $@
%.o: %.cc
	$(CC) $(CC_FLAGS) -c $< -o $@
%.ptx: %.cu
	$(NVCC) $(CU_FLAGS) --ptx $< -o $@

# Compile the assignment code
assign2_1: assign2_1.o $(CU_OBJECTS) $(CC_OBJECTS)
	$(NVCC) $^ -o $@

# Compile the vector-add program
vector-add: vector-add.o timer.o
	$(NVCC) $^ -o $@

# Compile to GPU-instructions only
ptx: $(CU_PTX) 

# Rule to clean all compiled files & results - always run before submitting or
#   when weird behaviour occurs!
clean:
	rm -f *.o *.ptx assign2_1 vector-add result.txt plot.png



##### RUN RULES #####

# Compiles & runs the code on the DAS
run: assign2_1
	prun -v -np 1 -native "-C TitanX" ./assign2_1 $(RUNARGS)
# Compiles & runs the code on a local, CUDA-supported machine
runlocal: assign2_1
	./assign2_1 $(RUNARGS)

# Plots the results
plot: result.txt
	gnuplot plot.gnp
	display plot.png
