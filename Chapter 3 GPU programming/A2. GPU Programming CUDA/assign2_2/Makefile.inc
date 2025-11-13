##### CONSTANTS #####

# Compilers we'll use
CC     := g++
NVCC   := nvcc
LINKER := g++

# Change this value to change the GPU compute capability your code will be
#   compiled against. Alternatively, compile with 'make COMPUTE=..." to make a
#   one-time change.
ifndef COMPUTE
COMPUTE = 52
endif

# Flags for each of the compilers
LDFLAGS         := -lm -lpthread
NVCCFLAGS	:= -arch compute_$(COMPUTE) -code sm_$(COMPUTE) --ptxas-options=-v

# Define some paths
PROJ_BASE	:= .
INCLUDES	:= -I"$(PROJ_BASE)"
LIBS		:= 
CUDA_LIBS	:= $(LIBS) -L"$(CUDA_INSTALL_PATH)/lib64"

# Depending on if debug is defined, add or remove extra flags
ifneq ($(debug), 1)
	CFLAGS		:= -O3 -g0 -w
	NVCCFLAGS	+= -O3
else
	CFLAGS		:= -O0 -g3 -Wall
	NVCCFLAGS	+= -O0
endif

# Arguments to run the file with (as i_max, t_max, block_size)
RUNARGS_CAESAR = 42
RUNARGS_CHECKSUM = original.data both
