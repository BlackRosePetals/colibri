#ifndef COLIBRI_DECODE_BATCH_H
#define COLIBRI_DECODE_BATCH_H

#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* `base` belongs to one sequence's KV state.  Keeping this arithmetic in a
 * model-independent seam makes ragged decode row ownership directly testable. */
static inline float *coli_kv_row(float *base, int position, int width)
{
    return base + (size_t)position * (size_t)width;
}

/* Per-token top-k emission cap: run_ablate_score's existing top-32 read-out
 * ceiling, adopted as the wire cap too (no named consumer asks for more). */
#define COLI_SUBMIT_TOPK_MAX 32

typedef struct {
    unsigned long long id, bytes, gbytes;
    int slot, max_tokens;
    float temperature, top_p;
    int logprobs;   /* requested per-token top-k count; 0 = channel off (opt-in) */
    int tok_ids;    /* 1 = payload is ASCII token ids, not raw prompt text */
} ColiSubmit;

/* Extension fields, 8th whitespace field onward: "key=value" tokens after the
 * full 7-field header, e.g. "SUBMIT 1 0 12 16 0 1 0 logprobs=5 ids=1".
 * A namespace instead of more positional fields: the next per-request knob
 * (a future seed, say) claims a key here rather than forcing another
 * field-count migration of this scarce wire budget. Unknown keys reject the
 * whole frame -- framing stays as unambiguous as the legacy arms'
 * trailing-field guard. An OLD engine rejects ANY extended header cleanly:
 * both legacy sscanf arms fail on the extra field (their trailing %c matches
 * the separating space) and the engine answers ERROR 0 BAD_REQUEST. */
static inline int coli_submit_ext(const char *p, ColiSubmit *s)
{
    int seen = 0;
    while (*p == ' ' || *p == '\t') p++;
    while (*p) {
        char key[16];
        unsigned long long val;
        int n = 0;
        if (sscanf(p, "%15[a-z_]=%llu%n", key, &val, &n) != 2 || n <= 0)
            return 0;
        if (p[n] && p[n] != ' ' && p[n] != '\t') return 0;
        if (!strcmp(key, "logprobs")) {          /* per-token top-k emission */
            if (val > COLI_SUBMIT_TOPK_MAX) return 0;
            s->logprobs = (int)val;
        } else if (!strcmp(key, "ids")) {        /* pre-tokenized prompt intake */
            if (val > 1) return 0;
            s->tok_ids = (int)val;
        } else {
            return 0;
        }
        seen = 1;
        p += n;
        while (*p == ' ' || *p == '\t') p++;
    }
    return seen;
}

/* Parse the textual header. The payload is read separately using `bytes`, so
 * it may contain newlines. Reject trailing fields to keep framing unambiguous.
 * Optional 7th field `gbytes`: length of a per-request grammar (raw GBNF, or a
 * JSON-Schema compiled engine-side) appended to the payload AFTER the prompt
 * bytes. 6-field headers remain valid (gbytes = 0). Fields past the 7th are
 * key=value extension tokens (coli_submit_ext above), only ever sent by a
 * server that knows this engine understands them. */
static inline int coli_submit_parse(const char *line, ColiSubmit *s)
{
    char tail;
    int ok = 0, base = 0;
    if (!line || !s) return 0;
    s->gbytes = 0;
    s->logprobs = 0;
    s->tok_ids = 0;
    if (sscanf(line, "SUBMIT %llu %d %llu %d %f %f %llu %c", &s->id, &s->slot,
               &s->bytes, &s->max_tokens, &s->temperature, &s->top_p,
               &s->gbytes, &tail) == 7) ok = 1;
    if (!ok) {
        s->gbytes = 0;
        if (sscanf(line, "SUBMIT %llu %d %llu %d %f %f %c", &s->id, &s->slot,
                   &s->bytes, &s->max_tokens, &s->temperature, &s->top_p,
                   &tail) == 6) ok = 1;
    }
    if (!ok) {
        s->gbytes = 0;
        if (sscanf(line, "SUBMIT %llu %d %llu %d %f %f %llu%n", &s->id,
                   &s->slot, &s->bytes, &s->max_tokens, &s->temperature,
                   &s->top_p, &s->gbytes, &base) == 7 && base > 0 &&
            coli_submit_ext(line + base, s)) ok = 1;
    }
    if (!ok) return 0;
    return s->id > 0 && s->bytes <= (16u << 20) && s->gbytes <= (1u << 20) &&
           s->slot >= 0 && s->max_tokens >= 1 &&
           isfinite(s->temperature) && isfinite(s->top_p) &&
           s->temperature >= 0 && s->temperature <= 2 &&
           s->top_p > 0 && s->top_p <= 1;
}

/* SUBMIT ids=1 payload: ASCII decimal token ids separated by whitespace,
 * parsed straight into the prompt buffer -- the pre-tokenized intake path
 * that bypasses tok_encode entirely (no detokenize/re-encode round trip, so
 * the ids the caller sent are exactly the ids the engine scores). Returns
 * the id count, or -1 on any malformed or out-of-range id. More than `cap`
 * ids reports `cap`: the caller parses with one spare slot and treats a full
 * buffer as overflow, mirroring tok_encode's stop-at-cap contract (#401). */
static inline int coli_ids_parse(const char *buf, size_t len, int *out,
                                 int cap, int vocab)
{
    size_t i = 0;
    int n = 0;
    if (!buf || !out || cap < 1 || vocab < 1) return -1;
    while (i < len) {
        while (i < len && (unsigned char)buf[i] <= ' ') i++;
        if (i >= len) break;
        if (n >= cap) return cap;            /* overflow: caller refuses loudly */
        {
            long v = 0;
            size_t d = i;
            while (i < len && buf[i] >= '0' && buf[i] <= '9' && v < vocab)
                v = v * 10 + (buf[i++] - '0');
            if (i == d || v >= vocab || (i < len && (unsigned char)buf[i] > ' '))
                return -1;
            out[n++] = (int)v;
        }
    }
    return n;
}

#endif
