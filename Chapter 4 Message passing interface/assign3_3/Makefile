# Makefile for Assignment 3.3 - MYMPI_Bcast

PROGNAME = assign3_3
SRCFILES = assign3_3.c mympi_bcast.c
OBJFILES = $(SRCFILES:.c=.o)

MPICC = mpicc
CFLAGS = -std=c99 -Wall -Wextra -O2

.PHONY: all run runlocal clean

all: $(PROGNAME)

$(PROGNAME): $(OBJFILES)
	$(MPICC) $(CFLAGS) -o $@ $^

%.o: %.c mympi_bcast.h
	$(MPICC) $(CFLAGS) -c $<

# Example: run on DAS-5 with 1 MPI processes
run: $(PROGNAME)
	prun -v -np 1 $(PROGNAME)

# Example: run locally with 1 MPI processes
runlocal: $(PROGNAME)
	mpirun -np 1 ./$(PROGNAME)

clean:
	rm -f $(PROGNAME) $(OBJFILES)
