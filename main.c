#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <malloc.h>

#include <xmmintrin.h>
#include <immintrin.h>

#define TEST_W 4096
#define TEST_H 4096
#define SAMPLES_NUM 100

/* provide the implementations of naive_transpose,
 * sse_transpose, sse_prefetch_transpose
 */

#include "impl.c"

typedef void (*transpose_cb)(int *src, int *dst, int w, int h);

/* verify the result of 4x4 matrix */
static void transpose_verify(transpose_cb transpose_impl)
{
    int testin[16] = { 0, 1,  2,  3,  4,  5,  6,  7,
                       8, 9, 10, 11, 12, 13, 14, 15
                     };
    int testout[16];
    int expected[16] = { 0, 4,  8, 12, 1, 5,  9, 13,
                         2, 6, 10, 14, 3, 7, 11, 15
                       };

    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++)
            printf(" %2d", testin[y * 4 + x]);
        printf("\n");
    }
    printf("\n");
    transpose_impl(testin, testout, 4, 4);
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++)
            printf(" %2d", testout[y * 4 + x]);
        printf("\n");
    }
    assert(0 == memcmp(testout, expected, 16 * sizeof(int)) &&
           "Verification fails");
}

static long diff_in_us(struct timespec t1, struct timespec t2)
{
    struct timespec diff;
    if (t2.tv_nsec-t1.tv_nsec < 0) {
        diff.tv_sec  = t2.tv_sec - t1.tv_sec - 1;
        diff.tv_nsec = t2.tv_nsec - t1.tv_nsec + 1000000000;
    } else {
        diff.tv_sec  = t2.tv_sec - t1.tv_sec;
        diff.tv_nsec = t2.tv_nsec - t1.tv_nsec;
    }
    return (diff.tv_sec * 1000000.0 + diff.tv_nsec / 1000.0);
}

static long sampling(int *src, int *out, transpose_cb transpose_impl)
{
    long result = 0;
    struct timespec start, end;
    for (int i = 0; i < SAMPLES_NUM; i++) {
        clock_gettime(CLOCK_REALTIME, &start);
        transpose_impl(src, out, TEST_W, TEST_H);
        clock_gettime(CLOCK_REALTIME, &end);
        result += diff_in_us(start, end);
    }
    return result / SAMPLES_NUM;
}

/* verify whether this two matrix is identical */
static int equal(int *a, int *b, int w, int h)
{
    for (int x = 0; x < w; x++)
        for (int y = 0; y < h; y++)
            if (*(a + x * h + y) != *(b + x * h + y)) {
                printf("x = %d y = %d\n", x, y);
                return -1;
            }
    return 1;
}

int main()
{
    long result = 0;
    int *src = (int *) memalign(16, sizeof(int) * TEST_W * TEST_H);
    int *out = (int *) memalign(16, sizeof(int) * TEST_W * TEST_H);
    int *out2 = (int *) memalign(16, sizeof(int) * TEST_W * TEST_H);

    srand(time(NULL));
    for (int y = 0; y < TEST_H; y++)
        for (int x = 0; x < TEST_W; x++)
            *(src + y * TEST_W + x) = rand();

#if SSE_PREFETCH
    transpose_verify(sse_prefetch_transpose);
    result = sampling(src, out, sse_prefetch_transpose);
    printf("sse prefetch: \t %ld us\n", result);
#endif

#if SSE
    transpose_verify(sse_transpose);
    result = sampling(src, out, sse_transpose);
    printf("sse: \t\t %ld us\n", result);
#endif

#if NAIVE
    transpose_verify(naive_transpose);
    result = sampling(src, out, naive_transpose);
    printf("naive: \t\t %ld us\n", result);
#endif

#if AVX
    avx_transpose(src, out, TEST_W, TEST_H);
    naive_transpose(src, out2, TEST_W, TEST_H);
    assert(1 == equal(out, out2, TEST_W, TEST_H && "Verification(AVX) fails"));
    result = sampling(src, out, avx_transpose);
    printf("avx: \t\t %ld us\n", result);
#endif

    free(src);
    free(out);
    free(out2);

    return 0;
}
