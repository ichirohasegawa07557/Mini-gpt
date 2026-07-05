#include "minigpt/rng.h"
#include <math.h>

void rng_seed(Rng *r, unsigned long long seed) { r->s = seed ? seed : 0x9E3779B97F4A7C15ULL; }

static unsigned long long next_u64(Rng *r) {
    unsigned long long x = r->s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    r->s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

double rng_uniform(Rng *r) { return (double)(next_u64(r) >> 11) / 9007199254740992.0; }

double rng_normal(Rng *r, double mean, double std) {
    double u1 = rng_uniform(r), u2 = rng_uniform(r);
    if (u1 < 1e-300) u1 = 1e-300;
    return mean + std * sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

int rng_randint(Rng *r, int n) { return n <= 0 ? 0 : (int)(next_u64(r) % (unsigned long long)n); }
