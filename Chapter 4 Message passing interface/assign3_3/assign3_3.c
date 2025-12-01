/* assign3_3.c
 *
 * Small test driver for MYMPI_Bcast.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "mympi_bcast.h"

int main(int argc, char *argv[])
{
    int rank, size;
    int root = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Optional: allow root to be set from command line */
    if (argc > 1) {
        root = atoi(argv[1]);
        if (root < 0 || root >= size) {
            if (rank == 0) {
                fprintf(stderr, "Invalid root %d (size = %d)\n", root, size);
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    /* Simple test buffer: each process starts with different value;
       after MYMPI_Bcast, everyone should hold root's value. */
    int value;

    if (rank == root) {
        value = 100 + root;  /* arbitrary value at root */
    } else {
        value = -1;          /* clearly different initial value */
    }

    if (rank == 0) {
        printf("Running MYMPI_Bcast with %d processes, root = %d\n",
               size, root);
    }

    MYMPI_Bcast(&value, 1, MPI_INT, root, MPI_COMM_WORLD);

    printf("Rank %d: value after broadcast = %d\n", rank, value);

    MPI_Finalize();
    return 0;
}
