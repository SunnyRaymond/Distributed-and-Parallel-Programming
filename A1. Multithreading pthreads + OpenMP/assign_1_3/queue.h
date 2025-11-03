#ifndef QUEUE_H
#define QUEUE_H

#include <pthread.h>
#include <stdbool.h>

typedef struct {
    int           *buf;
    int            cap;
    int            head;
    int            tail;
    int            count;
    pthread_mutex_t m;
    pthread_cond_t  not_full;
    pthread_cond_t  not_empty;
    bool           closed;   /* for graceful stop/broadcast */
} queue_t;

/* Initialize/destroy a bounded queue with capacity 'cap' */
int  q_init(queue_t *q, int cap);
void q_destroy(queue_t *q);

/* Blocking push/pop (return 0 on success, nonzero on closed) */
int  q_push(queue_t *q, int x);
int  q_pop(queue_t *q, int *out);

/* Mark queue as closed and wake up any waiters */
void q_close(queue_t *q);

#endif /* QUEUE_H */
