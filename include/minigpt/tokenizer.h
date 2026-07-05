#ifndef MINIGPT_TOKENIZER_H
#define MINIGPT_TOKENIZER_H

/* Character-level tokenizer — mirrors CharTokenizer (src/data.py). */
#define TOK_MAX_VOCAB 256

typedef struct {
    int  vocab_size;
    int  stoi[TOK_MAX_VOCAB];              /* byte -> id (-1 if absent) */
    unsigned char itos[TOK_MAX_VOCAB];     /* id -> byte */
} CharTokenizer;

void tokenizer_from_text(CharTokenizer *t, const char *text, long n);
int  tokenizer_encode(const CharTokenizer *t, const char *text, long n, int *ids);
void tokenizer_decode(const CharTokenizer *t, const int *ids, long n, char *out);

#endif
