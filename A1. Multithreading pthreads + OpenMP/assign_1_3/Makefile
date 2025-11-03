# Assignment 1.3 – Sieve of Eratosthenes (pthreads pipeline)

PROGNAME = assign1_3
SRCFILES = assign1_3.c queue.c
TARNAME  = assign1_3.tgz

# Optional run args:
#   COUNT: how many primes to print before exiting (0 = infinite)
#   QCAP : bounded queue capacity
# Example: make runlocal RUNARGS="1000 1024"
RUNARGS = 100 1024

CC = gcc
IMAGEVIEW = display

WARNFLAGS = -Wall -Werror-implicit-function-declaration -Wshadow \
            -Wstrict-prototypes -pedantic-errors
CFLAGS = -std=c99 -ggdb -O2 $(WARNFLAGS) -D_POSIX_C_SOURCE=200112
LFLAGS = -lm -lrt -lpthread

OBJFILES = $(patsubst %.c,%.o,$(SRCFILES))

.PHONY: all run runlocal clean dist

all: $(PROGNAME)

$(PROGNAME): $(OBJFILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

%.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<

run: $(PROGNAME)
	# Replace 'prun' invocation with your DAS-5 runner if needed
	prun -v -np 1 $(PROGNAME) $(RUNARGS)

runlocal: $(PROGNAME)
	./$(PROGNAME) $(RUNARGS)

dist:
	tar cvzf $(TARNAME) Makefile *.c *.h

clean:
	rm -fv $(PROGNAME) $(OBJFILES) $(TARNAME)
