/*
 * Assignment 1.3 – Sieve of Eratosthenes (pthreads pipeline)
 *
 * Usage:
 *   ./assign1_3 [COUNT] [QCAP]
 *     COUNT = number of primes to print before exiting (0 = infinite; Ctrl-C to stop)
 *     QCAP  = bounded queue capacity (default 1024)
 *
 * Design:
 *   - One generator thread produces natural numbers starting at 2 into a bounded queue.
 *   - A pipeline of filter threads: each filter reads from its inbound queue.
 *       * First value seen is a prime P -> print it.
 *       * Subsequent values: forward only those not divisible by P to its outbound queue.
 *       * The first time it needs to forward, it creates the outbound queue and spawns
 *         the next filter thread, using that queue as child’s input.
 *   - Communication only via bounded queues (mutex + condvars). No global locks on hot path.
 *   - Optional COUNT lets us stop after N primes for experiments; otherwise run unbounded.
 */

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <stdatomic.h>
#include <time.h>
#include <pthread.h>
#include "queue.h"

#define DEFAULT_QCAP 1024

/* Global stop control: set by reaching COUNT or SIGINT */
static volatile sig_atomic_t g_stop = 0;

/* For optional "print at most COUNT primes" */
static atomic_long g_printed = 0;
static long g_count_target = 0;

/* Serialize stdout to avoid interleaving */
static pthread_mutex_t g_print_mtx = PTHREAD_MUTEX_INITIALIZER;

/* Simple timing helper */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* SIGINT handler: request stop */
static void on_sigint(int signo) {
    (void)signo;
    g_stop = 1;
}

/* ------------------- Generator thread ------------------- */

typedef struct {
    queue_t *out_q;
} gen_args_t;

static void* generator_thread(void *arg) {
    gen_args_t *ga = (gen_args_t*)arg;
    int n = 2;
    while (!g_stop) {
        if (q_push(ga->out_q, n) != 0) break; /* closed */
        n++;
    }
    /* Close downstream to wake pipeline */
    q_close(ga->out_q);
    return NULL;
}

/* ------------------- Filter thread ------------------- */

typedef struct filter_args_s {
    queue_t *in_q;
    int      qcap;
} filter_args_t;

static void* filter_thread(void *arg) {
    filter_args_t *fa = (filter_args_t*)arg;

    int prime = 0;
    queue_t *out_q = NULL;
    pthread_t child_tid;
    int child_started = 0;

    for (;;) {
        if (g_stop) break;
        int x;
        if (q_pop(fa->in_q, &x) != 0) {
            /* Upstream closed & empty -> pass downstream close and exit */
            if (out_q) q_close(out_q);
            break;
        }

        if (prime == 0) {
            /* First number is the prime for this stage */
            prime = x;

            /* Print it */
            pthread_mutex_lock(&g_print_mtx);
            printf("%d\n", prime);
            fflush(stdout);
            pthread_mutex_unlock(&g_print_mtx);

            /* Optional bounded run: stop after g_count_target primes */
            if (g_count_target > 0) {
                long printed = atomic_fetch_add_explicit(&g_printed, 1, memory_order_relaxed) + 1;
                if (printed >= g_count_target) {
                    g_stop = 1;
                    /* Close my inbound (so upstream can wind down) and any outbound */
                    q_close(fa->in_q);
                    if (out_q) q_close(out_q);
                    break;
                }
            }
            continue;
        }

        /* Filter out multiples of 'prime' */
        if (x % prime != 0) {
            /* Lazily create child stage on first forward */
            if (!child_started) {
                out_q = (queue_t*)malloc(sizeof(queue_t));
                if (!out_q || q_init(out_q, fa->qcap) != 0) {
                    /* On failure, abort gracefully */
                    g_stop = 1;
                    if (out_q) free(out_q);
                    break;
                }
                filter_args_t *child_args = (filter_args_t*)malloc(sizeof(filter_args_t));
                if (!child_args) {
                    g_stop = 1;
                    q_close(out_q);
                    q_destroy(out_q);
                    free(out_q);
                    break;
                }
                child_args->in_q = out_q;
                child_args->qcap = fa->qcap;
                if (pthread_create(&child_tid, NULL, filter_thread, (void*)child_args) != 0) {
                    g_stop = 1;
                    q_close(out_q);
                    q_destroy(out_q);
                    free(out_q);
                    free(child_args);
                    break;
                }
                child_started = 1;
            }
            if (q_push(out_q, x) != 0) {
                /* Downstream closed */
                break;
            }
        }
    }

    /* Cleanup */
    if (child_started) {
        /* Close and let child drain/exit; not strictly necessary on infinite run */
        q_close(out_q);
        /* We won't join the child (pipeline can be long/infinite). */
    }
    q_close(fa->in_q);  /* in case upstream is blocked on push */
    free(fa);

    return NULL;
}

/* ------------------- main ------------------- */

int main(int argc, char **argv) {
    /* Parse args: COUNT QCAP */
    long count = 0;
    int qcap = DEFAULT_QCAP;
    if (argc >= 2) count = strtol(argv[1], NULL, 10);
    if (argc >= 3) qcap  = (int)strtol(argv[2], NULL, 10);
    if (qcap <= 0) qcap = DEFAULT_QCAP;

    g_count_target = count;  /* 0 means "infinite" */

    /* Set up SIGINT for graceful-ish stop */
    struct sigaction sa;
    sa.sa_handler = on_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);

    /* Create the first queue in the pipeline */
    queue_t *q0 = (queue_t*)malloc(sizeof(queue_t));
    if (!q0 || q_init(q0, qcap) != 0) {
        fprintf(stderr, "Failed to initialize queue\n");
        return EXIT_FAILURE;
    }

    /* Start generator */
    pthread_t gen_tid;
    gen_args_t ga = { .out_q = q0 };
    if (pthread_create(&gen_tid, NULL, generator_thread, &ga) != 0) {
        fprintf(stderr, "Failed to start generator\n");
        return EXIT_FAILURE;
    }

    /* Start first filter that reads from q0 (it will print '2', then build pipeline) */
    pthread_t filt_tid;
    filter_args_t *fa0 = (filter_args_t*)malloc(sizeof(filter_args_t));
    if (!fa0) {
        fprintf(stderr, "Failed to alloc filter args\n");
        g_stop = 1; q_close(q0);
        pthread_join(gen_tid, NULL);
        q_destroy(q0); free(q0);
        return EXIT_FAILURE;
    }
    fa0->in_q = q0;
    fa0->qcap = qcap;
    if (pthread_create(&filt_tid, NULL, filter_thread, (void*)fa0) != 0) {
        fprintf(stderr, "Failed to start first filter\n");
        g_stop = 1; q_close(q0);
        pthread_join(gen_tid, NULL);
        q_destroy(q0); free(q0);
        free(fa0);
        return EXIT_FAILURE;
    }

    /* If COUNT>0, time the run; otherwise we just run until Ctrl-C */
    double t0 = 0.0;
    if (g_count_target > 0) t0 = now_sec();

    /* Wait until we hit g_count_target (if any) or SIGINT sets g_stop. */
    if (g_count_target > 0) {
        /* Busy-waiting a bit is fine here; could also use a condvar with more plumbing. */
        while (!g_stop) {
            struct timespec ts = { .tv_sec = 0, .tv_nsec = 50 * 1000 * 1000 }; /* 50ms */
            nanosleep(&ts, NULL);
        }
    } else {
        /* For infinite mode, just join the generator; Ctrl-C will break things out. */
        pthread_join(gen_tid, NULL);
    }

    /* If we stopped due to COUNT, report timing. */
    if (g_count_target > 0) {
        double t1 = now_sec();
        long printed = atomic_load_explicit(&g_printed, memory_order_relaxed);
        fprintf(stderr, "Printed %ld primes in %.6f seconds (%.3f us/prime)\n",
                printed, t1 - t0, 1e6 * (t1 - t0) / (printed ? printed : 1));
    }

    /* Proactively close head queue (wakes pipeline) and wait generator */
    q_close(q0);
    pthread_join(gen_tid, NULL);

    /* Cleanup head queue */
    q_destroy(q0);
    free(q0);

    /* We do not join all filter descendants (the chain can be long); process will exit. */
    return EXIT_SUCCESS;
}
