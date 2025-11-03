#include "queue.h"
#include <stdlib.h>

int q_init(queue_t *q, int cap) {
    q->buf   = (int*)malloc(sizeof(int) * cap);
    if (!q->buf) return -1;
    q->cap   = cap;
    q->head  = 0;
    q->tail  = 0;
    q->count = 0;
    q->closed = false;
    pthread_mutex_init(&q->m, NULL);
    pthread_cond_init(&q->not_full, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    return 0;
}

void q_destroy(queue_t *q) {
    if (!q) return;
    free(q->buf);
    pthread_mutex_destroy(&q->m);
    pthread_cond_destroy(&q->not_full);
    pthread_cond_destroy(&q->not_empty);
}

void q_close(queue_t *q) {
    pthread_mutex_lock(&q->m);
    q->closed = true;
    pthread_cond_broadcast(&q->not_full);
    pthread_cond_broadcast(&q->not_empty);
    pthread_mutex_unlock(&q->m);
}

int q_push(queue_t *q, int x) {
    pthread_mutex_lock(&q->m);
    while (q->count == q->cap && !q->closed) {
        pthread_cond_wait(&q->not_full, &q->m);
    }
    if (q->closed) { pthread_mutex_unlock(&q->m); return -1; }
    q->buf[q->tail] = x;
    q->tail = (q->tail + 1) % q->cap;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->m);
    return 0;
}

int q_pop(queue_t *q, int *out) {
    pthread_mutex_lock(&q->m);
    while (q->count == 0 && !q->closed) {
        pthread_cond_wait(&q->not_empty, &q->m);
    }
    if (q->count == 0 && q->closed) {
        pthread_mutex_unlock(&q->m);
        return -1; /* closed and empty */
    }
    *out = q->buf[q->head];
    q->head = (q->head + 1) % q->cap;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->m);
    return 0;
}
