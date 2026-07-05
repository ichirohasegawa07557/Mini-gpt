#include "minigpt/generate.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* mirrors MiniGPT.generate: crop to block_size, take last-position logits,
 * temperature, optional top-k, softmax, multinomial sample. */
long generate_text(MiniGPT *m, const CharTokenizer *tok, Rng *rng,
                   const char *prompt, int max_new_tokens,
                   double temperature, int top_k,
                   char *out, long out_cap) {
    int T = m->cfg.block_size, V = m->cfg.vocab_size;
    long plen = (long)strlen(prompt), total, step;
    int *seq = (int *)malloc(sizeof(int) * (plen + max_new_tokens + 1));
    double *logits = (double *)malloc(sizeof(double) * (long)T * V);
    double *p = (double *)malloc(sizeof(double) * V);

    if (tokenizer_encode(tok, prompt, plen, seq) != 0) {
        fprintf(stderr, "prompt contains characters outside the corpus vocabulary\n");
        free(seq); free(logits); free(p);
        out[0] = '\0';
        return -1;
    }
    total = plen;
    for (step = 0; step < max_new_tokens; ++step) {
        long start = total > T ? total - T : 0;
        int Tc = (int)(total - start), v, next;
        model_forward(m, seq + start, Tc, logits);
        double *row = logits + (long)(Tc - 1) * V;

        if (temperature <= 0.0) {                    /* greedy argmax */
            next = 0;
            for (v = 1; v < V; ++v) if (row[v] > row[next]) next = v;
        } else {
            double thresh = -1e300, mx, denom = 0, r, acc = 0;
            if (top_k > 0 && top_k < V) {            /* k-th largest as threshold */
                double *tmp = (double *)malloc(sizeof(double) * V);
                int a, bnd;
                memcpy(tmp, row, sizeof(double) * V);
                for (a = 0; a < top_k; ++a) {        /* partial selection sort */
                    int mxi = a;
                    for (bnd = a + 1; bnd < V; ++bnd) if (tmp[bnd] > tmp[mxi]) mxi = bnd;
                    { double sw = tmp[a]; tmp[a] = tmp[mxi]; tmp[mxi] = sw; }
                }
                thresh = tmp[top_k - 1];
                free(tmp);
            }
            mx = -1e300;
            for (v = 0; v < V; ++v) {
                p[v] = (top_k > 0 && row[v] < thresh) ? -1e300 : row[v] / temperature;
                if (p[v] > mx) mx = p[v];
            }
            for (v = 0; v < V; ++v) { p[v] = exp(p[v] - mx); denom += p[v]; }
            r = rng_uniform(rng) * denom;
            next = V - 1;
            for (v = 0; v < V; ++v) { acc += p[v]; if (r <= acc) { next = v; break; } }
        }
        seq[total++] = next;
    }
    if (total >= out_cap) total = out_cap - 1;
    tokenizer_decode(tok, seq, total, out);
    free(seq); free(logits); free(p);
    return total;
}
