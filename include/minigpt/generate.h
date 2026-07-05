#ifndef MINIGPT_GENERATE_H
#define MINIGPT_GENERATE_H

#include "minigpt/model.h"

/* Autoregressive sampling with temperature and top-k,
 * mirrors MiniGPT.generate (src/model.py).
 * temperature <= 0 selects greedy argmax. top_k <= 0 disables top-k. */
long generate_text(MiniGPT *m, const CharTokenizer *tok, Rng *rng,
                   const char *prompt, int max_new_tokens,
                   double temperature, int top_k,
                   char *out, long out_cap);

#endif
