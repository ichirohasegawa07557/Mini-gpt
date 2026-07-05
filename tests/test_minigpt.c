#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "minigpt/model.h"
#include "minigpt/train.h"
#include "minigpt/generate.h"

static int failures = 0;
#define CHECK(cond, name) do { \
    if (cond) printf("PASS %s\n", name); \
    else { printf("FAIL %s\n", name); failures++; } } while (0)

/* mirrors test_tokenizer_roundtrip */
static void test_tokenizer(void) {
    CharTokenizer t;
    const char *s = "hello world";
    int ids[16]; char back[16];
    tokenizer_from_text(&t, s, (long)strlen(s));
    tokenizer_encode(&t, "hello", 5, ids);
    tokenizer_decode(&t, ids, 5, back);
    CHECK(strcmp(back, "hello") == 0, "tokenizer roundtrip");
    CHECK(t.vocab_size == 8, "tokenizer vocab size");  /* ' ','d','e','h','l','o','r','w' */
}

/* mirrors test_model_forward_and_loss */
static void test_forward_shapes(void) {
    GPTConfig g = {20, 8, 1, 2, 16};
    Rng rng; MiniGPT m;
    int x[8], y[8], i, ok = 1;
    double logits[8 * 20], loss;
    rng_seed(&rng, 7);
    model_init(&m, g, &rng);
    for (i = 0; i < 8; ++i) { x[i] = rng_randint(&rng, 20); y[i] = rng_randint(&rng, 20); }
    model_forward(&m, x, 8, logits);
    for (i = 0; i < 8 * 20; ++i) if (!isfinite(logits[i])) ok = 0;
    loss = model_loss_only(&m, x, y, 8);
    CHECK(ok, "forward logits finite");
    CHECK(isfinite(loss) && loss > 0, "loss finite and positive");
    /* untrained model on uniform targets should sit near ln(20) */
    CHECK(fabs(loss - log(20.0)) < 0.5, "initial loss near ln(vocab)");
    model_free(&m);
}

/* THE key correctness test: analytic grads vs central finite differences.
 * This is what proves the hand-written backward pass has no errors. */
static void test_gradient_check(void) {
    GPTConfig g = {11, 6, 1, 2, 8};
    Rng rng; MiniGPT m;
    int x[6], y[6], i, k, bad = 0;
    double eps = 1e-5, max_rel = 0;
    rng_seed(&rng, 99);
    model_init(&m, g, &rng);
    for (i = 0; i < 6; ++i) { x[i] = rng_randint(&rng, 11); y[i] = rng_randint(&rng, 11); }
    model_zero_grads(&m);
    model_loss_and_backward(&m, x, y, 6, 1);
    for (k = 0; k < 60; ++k) {                    /* 60 random parameters */
        long pi = (long)(rng_uniform(&rng) * m.n_params);
        double orig = m.params[pi], lp, lm, num, ana, rel;
        m.params[pi] = orig + eps; lp = model_loss_only(&m, x, y, 6);
        m.params[pi] = orig - eps; lm = model_loss_only(&m, x, y, 6);
        m.params[pi] = orig;
        num = (lp - lm) / (2 * eps);
        ana = m.grads[pi];
        rel = fabs(num - ana) / (fabs(num) + fabs(ana) + 1e-8);
        if (rel > max_rel) max_rel = rel;
        if (rel > 1e-4 && fabs(num - ana) > 1e-7) bad++;
    }
    printf("     gradient check max relative error: %.2e\n", max_rel);
    CHECK(bad == 0, "backward matches finite differences (60 params)");
    model_free(&m);
}

/* a few AdamW steps on one batch must reduce the loss */
static void test_loss_decreases(void) {
    GPTConfig g = {11, 6, 1, 2, 8};
    Rng rng; MiniGPT m; AdamW opt;
    int x[6], y[6], i, s;
    double l0, l1;
    rng_seed(&rng, 5);
    model_init(&m, g, &rng);
    adamw_init(&opt, m.n_params, 1e-2, 0.0);
    for (i = 0; i < 6; ++i) { x[i] = rng_randint(&rng, 11); y[i] = rng_randint(&rng, 11); }
    l0 = model_loss_only(&m, x, y, 6);
    for (s = 0; s < 30; ++s) {
        model_zero_grads(&m);
        model_loss_and_backward(&m, x, y, 6, 1);
        grad_clip_global_norm(m.grads, m.n_params, 1.0);
        adamw_step(&opt, m.params, m.grads);
    }
    l1 = model_loss_only(&m, x, y, 6);
    printf("     loss %.4f -> %.4f after 30 AdamW steps\n", l0, l1);
    CHECK(l1 < l0 * 0.5, "training reduces loss");
    adamw_free(&opt); model_free(&m);
}

/* mirrors test_generate_extends_sequence */
static void test_generate(void) {
    CharTokenizer t;
    const char *corpus = "abcdefgabcdefgabcdefg";
    GPTConfig g = {0, 8, 1, 2, 16};
    Rng rng; MiniGPT m;
    char out[64];
    long n;
    tokenizer_from_text(&t, corpus, (long)strlen(corpus));
    g.vocab_size = t.vocab_size;
    rng_seed(&rng, 3);
    model_init(&m, g, &rng);
    n = generate_text(&m, &t, &rng, "abcd", 5, 1.0, 3, out, sizeof(out));
    CHECK(n == 9 && strlen(out) == 9, "generate extends sequence by max_new_tokens");
    CHECK(strncmp(out, "abcd", 4) == 0, "generate preserves the prompt");
    model_free(&m);
}

/* attention rows must be a valid causal distribution */
static void test_attention_causal(void) {
    GPTConfig g = {11, 6, 1, 2, 8};
    Rng rng; MiniGPT m;
    int x[6], i, t, t2, ok = 1;
    double logits[6 * 11];
    rng_seed(&rng, 21);
    model_init(&m, g, &rng);
    for (i = 0; i < 6; ++i) x[i] = rng_randint(&rng, 11);
    model_forward(&m, x, 6, logits);
    for (t = 0; t < 6; ++t) {
        double s = 0;
        for (t2 = 0; t2 < 6; ++t2) {
            double a = m.last_att[(long)t * g.block_size + t2];
            if (t2 > t && a != 0.0) ok = 0;      /* causal: no future attention */
            s += a;
        }
        if (fabs(s - 1.0) > 1e-9) ok = 0;         /* rows sum to 1 */
    }
    CHECK(ok, "attention is causal and rows sum to 1");
    model_free(&m);
}

int main(void) {
    test_tokenizer();
    test_forward_shapes();
    test_attention_causal();
    test_gradient_check();
    test_loss_decreases();
    test_generate();
    if (failures == 0) { printf("all tests passed\n"); return 0; }
    printf("%d test(s) failed\n", failures);
    return 1;
}
