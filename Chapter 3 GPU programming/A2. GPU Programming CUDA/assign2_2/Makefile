##### CONSTANTS #####

# Change to enter debug mode and set different flags (see Makefile.inc)
debug		:= 0
# Includes another file - i.e., copies its contents right here
include		./Makefile.inc

# Define the library sources we'll need for compilation. If you create any new
#   .cc or .cu files used in the assignment, add them here
CU_SOURCES	= 
CC_SOURCES	= timer.cc file.cc

# Create paths to their relevant object files
CU_OBJECTS	= $(CU_SOURCES:%.cu=$(PROJ_BASE)/%.o)
CU_PTX		= $(CU_SOURCES:%.cu=$(PROJ_BASE)/%.ptx)
CC_OBJECTS	= $(CC_SOURCES:%.cc=$(PROJ_BASE)/%.o)



##### COMPILE RULES #####

# Mark certain rules as 'phony', i.e., not tied to actual files
.PHONY: default ptx run runlocal all clean
# Define the default rule to compile
default: all
# Rule to compile all 'major' rules
all: caesar checksum

# Compile objects; either as CUDA objects (.cu), C++ objects (.cc) or GPU assembly
$(PROJ_BASE)/%.o: $(PROJ_BASE)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@
$(PROJ_BASE)/%.o: $(PROJ_BASE)/%.cc
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
$(PROJ_BASE)/%.ptx: $(PROJ_BASE)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) --ptx $< -o $@

# Compile the encryption / decryption function
caesar: $(PROJ_BASE)/caesar.o $(CU_OBJECTS) $(CC_OBJECTS)
	$(NVCC) $^ -o $@ $(INCLUDES) $(CUDA_LIBS) $(NVCCFLAGS) $(LDFLAGS)

# Compile the checksum function
checksum: $(PROJ_BASE)/checksum.o $(CU_OBJECTS) $(CC_OBJECTS)
	$(NVCC) $^ -o $@ $(INCLUDES) $(CUDA_LIBS) $(NVCCFLAGS) $(LDFLAGS)

# Compile to GPU-instructions only
ptx: $(CU_PTX) 

# Rule to clean all compiled files & results - always run before submitting or
#   when weird behaviour occurs!
clean:
	rm -f $(PROJ_BASE)/*.o $(PROJ_BASE)/*.ptx $(PROJ_BASE)/caesar $(PROJ_BASE)/checksum
	rm -f sequential.data cuda.data sequential_recovered.data recovered.data



##### RUN RULES #####

# Compiles & runs the code on the DAS
run: $(PROJ_BASE)/caesar $(PROJ_BASE)/checksum
	prun -v -np 1 -native "-C TitanX" $(PROJ_BASE)/caesar $(RUNARGS_CAESAR)
	prun -v -np 1 -native "-C TitanX" $(PROJ_BASE)/checksum $(RUNARGS_CHECKSUM)
# Compiles & runs the code on a local, CUDA-supported machine
runlocal: $(PROJ_BASE)/caesar $(PROJ_BASE)/checksum
	$(PROJ_BASE)/caesar $(RUNARGS_CAESAR)
	$(PROJ_BASE)/checksum $(RUNARGS_CHECKSUM)
