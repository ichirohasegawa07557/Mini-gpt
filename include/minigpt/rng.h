#ifndef MINIGPT_RNG_H
#define MINIGPT_RNG_H

/* xorshift64* PRNG: deterministic, seedable, no rand() from libc. */
typedef struct { unsigned long long s; } Rng;

void   rng_seed(Rng *r, unsigned long long seed);
double rng_uniform(Rng *r);              /* [0,1) */
double rng_normal(Rng *r, double mean, double std); /* Box-Muller */
int    rng_randint(Rng *r, int n);       /* [0,n) */

#endif
