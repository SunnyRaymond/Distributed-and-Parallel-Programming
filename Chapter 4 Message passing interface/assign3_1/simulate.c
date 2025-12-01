/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>          /* MPI header */

#include "simulate.h"


/* Add any global variables you may need. */


/* Add any functions you may need (like a worker) here. */


/*
 * Executes the entire simulation.
 *
 * Implement your code here.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int i_max, const int t_max, double *old_array,
        double *current_array, double *next_array)
{
    int rank, size;

    /* Must be the first MPI call */
    MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---- domain decomposition: counts / displacements ---- */
    int *counts  = (int *) malloc(size * sizeof(int));
    int *displs  = (int *) malloc(size * sizeof(int));
    if (!counts || !displs) {
        fprintf(stderr, "Rank %d: failed to allocate counts/displs\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int base = i_max / size;
    int rem  = i_max % size;
    for (int p = 0; p < size; ++p) {
        counts[p] = base + (p < rem ? 1 : 0);
    }
    displs[0] = 0;
    for (int p = 1; p < size; ++p) {
        displs[p] = displs[p - 1] + counts[p - 1];
    }

    int local_n   = counts[rank];   /* number of points owned by this rank   */
    int global_lo = displs[rank];   /* global index of local point 1         */

    /* local arrays with two halo cells */
    double *old_local   = (double *) malloc((local_n + 2) * sizeof(double));
    double *cur_local   = (double *) malloc((local_n + 2) * sizeof(double));
    double *next_local  = (double *) malloc((local_n + 2) * sizeof(double));
    if (!old_local || !cur_local || !next_local) {
        fprintf(stderr, "Rank %d: failed to allocate local buffers\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* ---- scatter initial data from rank 0 ---- */
    double *sendbuf_old = (rank == 0) ? old_array     : NULL;
    double *sendbuf_cur = (rank == 0) ? current_array : NULL;

    MPI_Scatterv(sendbuf_old, counts, displs, MPI_DOUBLE,
                 &old_local[1], local_n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(sendbuf_cur, counts, displs, MPI_DOUBLE,
                 &cur_local[1], local_n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    /* set halo cells to 0 initially (safe default) */
    old_local[0]          = 0.0;
    old_local[local_n+1]  = 0.0;
    cur_local[0]          = 0.0;
    cur_local[local_n+1]  = 0.0;

    /* neighbours in the 1D process grid */
    int left  = (rank == 0)        ? MPI_PROC_NULL : rank - 1;
    int right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    const double c = 0.15;   /* same wave constant as in previous labs */

    /* ---- time stepping ---- */
    for (int t = 1; t <= t_max; ++t) {

        /* exchange halo values with neighbours (blocking) */
        MPI_Sendrecv(&cur_local[1],        1, MPI_DOUBLE, left,  0,
                     &cur_local[local_n+1], 1, MPI_DOUBLE, right, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&cur_local[local_n],  1, MPI_DOUBLE, right, 1,
                     &cur_local[0],        1, MPI_DOUBLE, left,  1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* update local points (global endpoints remain fixed to 0) */
        for (int i = 1; i <= local_n; ++i) {
            int global_i = global_lo + (i - 1);

            if (global_i == 0 || global_i == i_max - 1) {
                /* fixed boundary conditions */
                next_local[i] = 0.0;
            } else {
                next_local[i] =
                    2.0 * cur_local[i] - old_local[i] +
                    c * (cur_local[i - 1] - 2.0 * cur_local[i] + cur_local[i + 1]);
            }
        }

        /* rotate buffers: old <- cur <- next */
        double *tmp = old_local;
        old_local   = cur_local;
        cur_local   = next_local;
        next_local  = tmp;
    }

    /* ---- write local results back into the full array on each rank ---- */
    for (int i = 1; i <= local_n; ++i) {
        int global_i = global_lo + (i - 1);
        current_array[global_i] = cur_local[i];
    }

    /* clean up local MPI-related buffers */
    free(old_local);
    free(cur_local);
    free(next_local);
    free(counts);
    free(displs);

    /* finalize MPI before returning to the framework */
    MPI_Finalize();

    /* every rank now has a full, correct current_array */
    return current_array;
}

