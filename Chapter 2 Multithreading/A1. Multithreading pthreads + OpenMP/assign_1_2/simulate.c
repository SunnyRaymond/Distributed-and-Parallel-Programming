/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>

#include "simulate.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Executes the entire simulation.
 *
 * Implement your code here.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * num_threads: how many threads to use
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int i_max, const int t_max, const int num_threads,
        double *old_array, double *current_array, double *next_array)
{
    const double c = 0.15;

    /* Trivial/edge cases */
    if (t_max <= 0 || i_max <= 0) {
        return current_array;
    }
    if (i_max <= 2) {
        for (int t = 0; t < t_max; ++t) {
            if (i_max >= 1) next_array[0] = 0.0;
            if (i_max == 2) next_array[1] = 0.0;
            double *tmp = old_array; old_array = current_array; current_array = next_array; next_array = tmp;
        }
        return current_array;
    }

#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    #pragma omp parallel
    {
        for (int t = 0; t < t_max; ++t) {
            /* Parallel interior update: i = 1 .. i_max-2 */
            #pragma omp for /* schedule(static) by default; vary in experiments */
            for (int i = 1; i < i_max - 1; ++i) {
                next_array[i] = 2.0 * current_array[i]
                              - old_array[i]
                              + c * (current_array[i - 1] - 2.0 * current_array[i] + current_array[i + 1]);
            }

            /* One thread sets fixed boundaries and rotates buffers */
            #pragma omp single
            {
                next_array[0] = 0.0;
                next_array[i_max - 1] = 0.0;

                double *tmp = old_array;
                old_array   = current_array;
                current_array = next_array;
                next_array = tmp;
            }
            /* Implicit barrier at end of 'single' ensures all see rotated buffers */
        }
    }
#else
    /* Fallback: sequential */
    for (int t = 0; t < t_max; ++t) {
        next_array[0] = 0.0;
        next_array[i_max - 1] = 0.0;
        for (int i = 1; i < i_max - 1; ++i) {
            next_array[i] = 2.0 * current_array[i]
                          - old_array[i]
                          + c * (current_array[i - 1] - 2.0 * current_array[i] + current_array[i + 1]);
        }
        double *tmp = old_array; old_array = current_array; current_array = next_array; next_array = tmp;
    }
#endif

    return current_array;
}

