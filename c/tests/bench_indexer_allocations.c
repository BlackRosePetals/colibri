/* Benchmark: Measure baseline vs persistent scratch buffer allocation for DeepSeek V4 indexer. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

typedef struct { float score; int index; } IndexScore;

// --- Baseline Implementation: Dynamic Malloc/Free per call ---
double indexer_select_batch_baseline(int batch, int need, int heads, int dimension, int max_count, int cols) {
    size_t qn = (size_t)heads * dimension;

    // 9 dynamic allocations
    volatile float *queries = (volatile float *)malloc((size_t)batch * qn * sizeof(float));
    volatile float *sq = (volatile float *)malloc((size_t)need * qn * sizeof(float));
    volatile float *head_weights = (volatile float *)malloc((size_t)need * heads * sizeof(float));
    volatile int *scounts = (volatile int *)malloc((size_t)need * sizeof(int));
    volatile int *stoken = (volatile int *)malloc((size_t)need * sizeof(int));
    volatile float *scores = (volatile float *)malloc((size_t)need * max_count * sizeof(float));
    volatile IndexScore *ranked = (volatile IndexScore *)malloc((size_t)max_count * sizeof(IndexScore));
    volatile uint8_t *scales = (volatile uint8_t *)malloc((size_t)dimension / 32);
    volatile float *qdq = (volatile float *)malloc((size_t)dimension * sizeof(float));

    // Optional GPU path allocations
    volatile float *xq = (volatile float *)malloc((size_t)need * cols * sizeof(float));
    volatile float *yq = (volatile float *)malloc((size_t)need * qn * sizeof(float));
    volatile uint8_t *xs = (volatile uint8_t *)malloc((size_t)need * (cols / 128));

    // Write to memory to prevent compiler DCE
    if (queries) queries[0] = 1.0f;
    if (sq) sq[0] = 2.0f;
    if (scores) scores[0] = 3.0f;
    if (xq) xq[0] = 4.0f;

    double sum = (queries ? queries[0] : 0) + (sq ? sq[0] : 0) + (scores ? scores[0] : 0) + (xq ? xq[0] : 0);

    // 12 dynamic frees
    free((void *)xs); free((void *)yq); free((void *)xq);
    free((void *)qdq); free((void *)scales); free((void *)ranked); free((void *)scores); free((void *)stoken);
    free((void *)scounts); free((void *)head_weights); free((void *)sq); free((void *)queries);

    return sum;
}

// --- Persistent Arena Scratch Buffer Implementation ---
typedef struct {
    void *buf;
    size_t cap;
} ScratchBuf;

static void ensure_capacity(ScratchBuf *s, size_t needed) {
    if (s->cap < needed) {
        s->cap = needed < 65536 ? 65536 : needed * 2;
        s->buf = realloc(s->buf, s->cap);
    }
}

static ScratchBuf g_scratch = {NULL, 0};

double indexer_select_batch_optimized(int batch, int need, int heads, int dimension, int max_count, int cols) {
    size_t qn = (size_t)heads * dimension;

    size_t sz_queries = (size_t)batch * qn * sizeof(float);
    size_t sz_sq = (size_t)need * qn * sizeof(float);
    size_t sz_head_weights = (size_t)need * heads * sizeof(float);
    size_t sz_scounts = (size_t)need * sizeof(int);
    size_t sz_stoken = (size_t)need * sizeof(int);
    size_t sz_scores = (size_t)need * max_count * sizeof(float);
    size_t sz_ranked = (size_t)max_count * sizeof(IndexScore);
    size_t sz_scales = (size_t)dimension / 32;
    size_t sz_qdq = (size_t)dimension * sizeof(float);
    size_t sz_xq = (size_t)need * cols * sizeof(float);
    size_t sz_yq = (size_t)need * qn * sizeof(float);
    size_t sz_xs = (size_t)need * (cols / 128);

    size_t total = sz_queries + sz_sq + sz_head_weights + sz_scounts + sz_stoken +
                   sz_scores + sz_ranked + sz_scales + sz_qdq + sz_xq + sz_yq + sz_xs;

    ensure_capacity(&g_scratch, total);

    char *ptr = (char *)g_scratch.buf;
    volatile float *queries = (volatile float *)ptr; ptr += sz_queries;
    volatile float *sq = (volatile float *)ptr; ptr += sz_sq;
    volatile float *head_weights = (volatile float *)ptr; ptr += sz_head_weights;
    volatile int *scounts = (volatile int *)ptr; ptr += sz_scounts;
    volatile int *stoken = (volatile int *)ptr; ptr += sz_stoken;
    volatile float *scores = (volatile float *)ptr; ptr += sz_scores;
    volatile IndexScore *ranked = (volatile IndexScore *)ptr; ptr += sz_ranked;
    volatile uint8_t *scales = (volatile uint8_t *)ptr; ptr += sz_scales;
    volatile float *qdq = (volatile float *)ptr; ptr += sz_qdq;
    volatile float *xq = (volatile float *)ptr; ptr += sz_xq;
    volatile float *yq = (volatile float *)ptr; ptr += sz_yq;
    volatile uint8_t *xs = (volatile uint8_t *)ptr;

    // Write & read memory to prevent compiler DCE
    queries[0] = 1.0f;
    sq[0] = 2.0f;
    scores[0] = 3.0f;
    xq[0] = 4.0f;

    (void)xs; (void)yq; (void)qdq; (void)scales; (void)ranked; (void)stoken; (void)scounts; (void)head_weights;
    return (double)(queries[0] + sq[0] + scores[0] + xq[0]);
}

int main(void) {
    const int batch = 32;
    const int need = 32;
    const int heads = 64;
    const int dimension = 128;
    const int max_count = 2048;
    const int cols = 2048;

    const int TRIALS = 10;
    const int REPEATS = 50000; // 50,000 calls per trial

    printf("[OK] Indexer allocation micro-benchmark initialized.\n\n");
    printf("--- Running 10-Trial Benchmark (%d indexer calls / trial) ---\n", REPEATS);

    double total_base_s = 0.0;
    double total_opt_s = 0.0;
    volatile double dummy = 0.0;

    for (int t = 0; t < TRIALS; t++) {
        // Measure baseline
        clock_t t0 = clock();
        for (int r = 0; r < REPEATS; r++) {
            dummy += indexer_select_batch_baseline(batch, need, heads, dimension, max_count, cols);
        }
        clock_t t1 = clock();
        double time_base = (double)(t1 - t0) / CLOCKS_PER_SEC;

        // Measure optimized
        t0 = clock();
        for (int r = 0; r < REPEATS; r++) {
            dummy += indexer_select_batch_optimized(batch, need, heads, dimension, max_count, cols);
        }
        t1 = clock();
        double time_opt = (double)(t1 - t0) / CLOCKS_PER_SEC;

        total_base_s += time_base;
        total_opt_s += time_opt;

        double speedup = ((time_base - time_opt) / time_base) * 100.0;
        printf("Trial %2d: Baseline = %.4f s | Optimized = %.4f s | Speedup = %.2fx (%.1f%% faster)\n",
               t + 1, time_base, time_opt, time_base / time_opt, speedup);
    }

    double avg_base = total_base_s / TRIALS;
    double avg_opt = total_opt_s / TRIALS;
    double avg_speedup = ((avg_base - avg_opt) / avg_base) * 100.0;

    printf("\n=== 10-TRIAL AVERAGE SUMMARY ===\n");
    printf("Average Baseline Time: %.4f s\n", avg_base);
    printf("Average Optimized Time: %.4f s\n", avg_opt);
    printf("Average Speedup:       %.2fx FASTER (%.1f%% latency reduction)\n",
           avg_base / avg_opt, avg_speedup);

    if (g_scratch.buf) free(g_scratch.buf);
    (void)dummy;
    return 0;
}
