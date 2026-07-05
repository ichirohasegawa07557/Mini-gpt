#include "minigpt/tokenizer.h"

/* sorted(set(text)) — mirrors CharTokenizer.from_text */
void tokenizer_from_text(CharTokenizer *t, const char *text, long n) {
    int seen[TOK_MAX_VOCAB] = {0};
    long i;
    for (i = 0; i < n; ++i) seen[(unsigned char)text[i]] = 1;
    t->vocab_size = 0;
    for (i = 0; i < TOK_MAX_VOCAB; ++i) t->stoi[i] = -1;
    for (i = 0; i < TOK_MAX_VOCAB; ++i) {   /* byte order == sorted order */
        if (seen[i]) {
            t->stoi[i] = t->vocab_size;
            t->itos[t->vocab_size] = (unsigned char)i;
            t->vocab_size++;
        }
    }
}

int tokenizer_encode(const CharTokenizer *t, const char *text, long n, int *ids) {
    long i;
    for (i = 0; i < n; ++i) {
        int id = t->stoi[(unsigned char)text[i]];
        if (id < 0) return -1;              /* unknown char: report, don't crash */
        ids[i] = id;
    }
    return 0;
}

void tokenizer_decode(const CharTokenizer *t, const int *ids, long n, char *out) {
    long i;
    for (i = 0; i < n; ++i) out[i] = (char)t->itos[ids[i]];
    out[n] = '\0';
}
