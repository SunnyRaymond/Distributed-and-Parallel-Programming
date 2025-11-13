/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>

#include "simulate.h"

/* Add any global variables you may need. */
#include <pthread.h>

static const double g_c = 0.15;

static int g_i_max = 0;
static int g_t_max = 0;
static int g_num_threads = 0;

static double *g_old = NULL;
static double *g_cur = NULL;
static double *g_next = NULL;

static pthread_barrier_t g_barrier;

typedef struct {
    int start;  // inclusive
    int end;    // exclusive
} thread_args_t;

/* Add any functions you may need (like a worker) here. */
static void *worker(void *arg)
{
    thread_args_t *a = (thread_args_t *)arg;

    for (int t = 0; t < g_t_max; ++t) {
        /* Compute interior points for this thread's chunk. */
        for (int i = a->start; i < a->end; ++i) {
            g_next[i] = 2.0 * g_cur[i]
                      - g_old[i]
                      + g_c * (g_cur[i - 1] - 2.0 * g_cur[i] + g_cur[i + 1]);
        }

        /* First barrier: wait for all threads (and main) to finish compute & boundary sets. */
        pthread_barrier_wait(&g_barrier);

        /* Main thread rotates buffers between the two barriers. */
        pthread_barrier_wait(&g_barrier);
    }

    return NULL;
}

/*
 * Executes the entire simulation.
 *
 * Implement your code here.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * num_threads: how many threads to use (excluding the main threads)
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int i_max, const int t_max, const int num_threads,
        double *old_array, double *current_array, double *next_array)
{
    /* Trivial/edge cases: nothing to do. */
    if (t_max <= 0 || i_max <= 0) {
        return current_array;
    }

    /* If there are no interior points, just enforce boundaries to 0 across steps. */
    if (i_max <= 2) {
        /* Only boundaries exist; they must stay zero. */
        for (int t = 0; t < t_max; ++t) {
            if (i_max >= 1) next_array[0] = 0.0;
            if (i_max == 2) next_array[1] = 0.0;
            /* Rotate old <- current, current <- next, next <- old */
            double *tmp = old_array;
            old_array = current_array;
            current_array = next_array;
            next_array = tmp;
        }
        return current_array;
    }

    /* Sequential fallback if num_threads <= 0: compute locally without pthreads. */
    if (num_threads <= 0) {
        for (int t = 0; t < t_max; ++t) {
            next_array[0] = 0.0;
            next_array[i_max - 1] = 0.0;
            for (int i = 1; i < i_max - 1; ++i) {
                next_array[i] = 2.0 * current_array[i]
                              - old_array[i]
                              + g_c * (current_array[i - 1] - 2.0 * current_array[i] + current_array[i + 1]);
            }
            double *tmp = old_array;
            old_array = current_array;
            current_array = next_array;
            next_array = tmp;
        }
        return current_array;
    }

    /* Setup globals for worker threads. */
    g_i_max = i_max;
    g_t_max = t_max;
    g_num_threads = num_threads;
    g_old = old_array;
    g_cur = current_array;
    g_next = next_array;

    /* Partition interior points [1, i_max-1) among threads in contiguous blocks. */
    const int interior = i_max - 2;               /* number of updatable points */
    const int base = interior / num_threads;
    const int rem  = interior % num_threads;

    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * num_threads);
    thread_args_t *args = (thread_args_t *)malloc(sizeof(thread_args_t) * num_threads);
    if (!threads || !args) {
        /* Fallback to sequential if allocation fails. */
        if (threads) free(threads);
        if (args) free(args);
        for (int t = 0; t < t_max; ++t) {
            g_next[0] = 0.0;
            g_next[i_max - 1] = 0.0;
            for (int i = 1; i < i_max - 1; ++i) {
                g_next[i] = 2.0 * g_cur[i]
                          - g_old[i]
                          + g_c * (g_cur[i - 1] - 2.0 * g_cur[i] + g_cur[i + 1]);
            }
            double *tmp = g_old; g_old = g_cur; g_cur = g_next; g_next = tmp;
        }
        return g_cur;
    }

    /* Initialize barrier for all workers + main thread. */
    pthread_barrier_init(&g_barrier, NULL, (unsigned)(num_threads + 1));

    /* Create threads with their assigned ranges. */
    int offset = 1; /* first interior index */
    for (int t = 0; t < num_threads; ++t) {
        int len = base + (t < rem ? 1 : 0);
        args[t].start = offset;
        args[t].end   = offset + len;
        offset += len;

        /* Threads with empty ranges are allowed (in case num_threads > interior). */
        pthread_create(&threads[t], NULL, worker, (void *)&args[t]);
    }

    /* Time-stepping loop: main thread sets boundaries, synchronizes, and rotates buffers. */
    for (int step = 0; step < t_max; ++step) {
        /* Fixed boundary conditions each step. */
        g_next[0] = 0.0;
        g_next[g_i_max - 1] = 0.0;

        /* First barrier: wait until all workers finished their chunk for this step. */
        pthread_barrier_wait(&g_barrier);

        /* Rotate the three buffers (old <- cur, cur <- next, next <- old). */
        double *tmp = g_old;
        g_old = g_cur;
        g_cur = g_next;
        g_next = tmp;

        /* Second barrier: let workers observe the new generation before continuing. */
        pthread_barrier_wait(&g_barrier);
    }

    /* Join and cleanup. */
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }
    pthread_barrier_destroy(&g_barrier);
    free(threads);
    free(args);

    /* You should return a pointer to the array with the final results. */
    return g_cur;
}
