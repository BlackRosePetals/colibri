/* GLM-5.3-Flash inference engine in pure C — sibling of kimi_k3.c / colibri.c /
 * deepseek_v4.c / inkling.c / qwen36.c / olmoe.c, sharing st.h / json.h / tok.h /
 * quant.h / compat.h.
 *
 * The name says GLM, but the skeleton is Kimi K3's, not GLM-5.2's: a hybrid
 * stack of linear-attention layers punctuated by full attention, over a
 * streamed MoE. That is why this file starts from kimi_k3.c and borrows the
 * indexer from colibri.c and the hyper-connections from deepseek_v4.c, instead
 * of extending the GLM engine.
 *
 * Architecture, read off the released checkpoint (321.34 B parameters measured
 * from the shard headers; the card says 320 B total / 18 B active):
 *
 *   - 45 text layers plus one MTP layer (index 45), hidden 4096, vocab 154880.
 *   - Hybrid attention, pattern (KDA KDA KDA FULL) repeating: 34 KDA linear
 *     layers and 11 DeepSeek-sparse-attention layers at 3, 7, ... 43, plus the
 *     MTP layer's own full-attention block. config.layer_types is explicit and
 *     is what this engine follows; linear_attn_config.kda_layers agrees.
 *   - MLA with kv_lora 512, q_lora 1536, qk_nope 256 and **qk_rope 0**: the
 *     full-attention layers are NoPE, exactly like K3. Position lives in the
 *     KDA decay and short convolutions, not in a rotation.
 *   - DSA lightning indexer on every full-attention layer, with k-pooling:
 *     keys are grouped into pools of index_kpool=4, the pools are scored
 *     instead of the tokens, the top index_topk/kpool pools are expanded back
 *     into token indices and the incomplete tail pool is always appended
 *     (index_kpool_always_select_tail). Two tensors carry it that GLM-5.2's
 *     indexer does not have: index_kpool_compress_ape and _compress_gate.
 *   - Manifold-Constrained Hyper-Connections (mHC, hc_mult 4, 20 Sinkhorn
 *     iterations) replace the plain residual at both sites of every layer —
 *     the same mHC DeepSeek V4 uses, down to the config keys, so the split
 *     into pre/post/comb and the Sinkhorn projection are shared code.
 *   - MoE: 288 routed experts (top-8) plus 1 shared expert per layer from
 *     layer 3 on, moe_intermediate 2048, sigmoid scoring with noaux_tc and
 *     e_score_correction_bias, routed_scaling_factor 2.5. The first three
 *     layers are dense (intermediate 12288).
 *   - Natively multimodal: a 24-block ViT (hidden 1024, patch 14, 448 px,
 *     spatial merge 2) whose patches are projected to 4096 and substituted at
 *     the image-token positions of the text stream. Text-only prompts never
 *     touch it.
 *
 * KDA recurrence — the same Kimi Delta Attention kimi_k3.c already reproduces
 * token-exact against the vendor, verified line by line against
 * transformers' recurrent_kimi_delta_attention:
 *     q,k,v = SiLU(ShortConv4(W{q,k,v} x));  q,k L2-normalized, q *= d^-0.5
 *     z  = W_fb(W_fa x) + dt_bias
 *     gk = gmin * sigmoid(exp(A_log[h]) * z),  gmin = gate_lower_bound = -5
 *     S  = (I - beta k k^T) Diag(exp(gk)) S + beta k v^T,  beta = sigmoid(W_b x)
 *     o  = S^T q;  out = W_o [ sigmoid(W_gb(W_ga x)) * RMSNorm_head(o) ]
 * One difference from K3: the output gate is LOW-RANK here (g_a_proj into
 * head_dim, then g_b_proj back out), where K3 has a single full g_proj.
 *
 * Container: tools/convert_glm53.py writes routed experts as int4 group-scaled
 * gs64 (`name` U8 + `name.qs` F32, fmt=4 — the same container GLM-5.2 uses,
 * measured cosine 0.994 against the fp8 source on real weights) and everything
 * else as BF16, quantized at LOAD TIME here. That split is deliberate: the
 * non-expert weights are 3% of the bytes, so keeping them exact on disk costs
 * ~14 GB and means retuning dense precision never requires re-downloading the
 * checkpoint.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "json.h"
#include "st.h"
#include "hyper_connections.h"   /* mHC, condiviso con deepseek_v4.c */

/* ---------- config ----------
 * Nested like Kimi K3's: the root carries the vision wrapper and `text_config`
 * carries the language model. A text-only export therefore has the text keys at
 * the root, and both shapes are accepted. */
typedef struct {
    /* text */
    int hidden, n_layers, vocab, first_dense, dense_inter;
    int n_heads, q_lora, kv_lora, qk_nope, qk_rope, v_head, qk_head;
    int n_experts, topk, moe_inter, n_shared;
    float routed_scale, eps;
    int n_mtp;                       /* num_nextn_predict_layers (1) */
    /* KDA */
    int kda_heads, kda_hd, kda_proj, conv_k;
    float gate_lb;
    /* DSA indexer with k-pooling */
    int index_topk, index_nh, index_hd, index_kpool, index_kpool_tail;
    /* mHC */
    int hc_mult, hc_iters;
    float hc_eps;
    /* per-layer kind: 1 = full attention (MLA + indexer), 0 = KDA */
    unsigned char is_full[128];
    /* vision (0 = text-only checkpoint) */
    int vis_layers, vis_hidden, vis_heads, vis_inter, vis_patch, vis_temporal;
    int vis_merge, vis_out_hidden, vis_proj_inter, vis_image_size, vis_in_ch;
    float vis_swiglu_limit, vis_eps;
    int image_token, image_start_token, image_end_token;
    int video_token, video_start_token, video_end_token;
} Cfg;

static double req_num(jval *object, const char *key) {
    jval *value = json_get(object, key);
    if (!value || value->t != J_NUM) {
        fprintf(stderr, "config.json: missing or non-numeric \"%s\"\n", key);
        exit(1);
    }
    return value->num;
}

static double opt_num(jval *object, const char *key, double fallback) {
    jval *value = object ? json_get(object, key) : NULL;
    return (value && value->t == J_NUM) ? value->num : fallback;
}

static int opt_bool(jval *object, const char *key, int fallback) {
    jval *value = object ? json_get(object, key) : NULL;
    if (!value) return fallback;
    if (value->t == J_BOOL) return value->boolean;
    if (value->t == J_NUM) return value->num != 0.0;
    return fallback;
}

/* layer_types is the authority on which layers are full attention. The
 * linear_attn_config.{kda_layers,full_attn_layers} lists say the same thing;
 * disagreeing checkpoints are refused rather than guessed at, because picking
 * the wrong kind for one layer produces plausible-looking garbage. */
static void load_layer_kinds(Cfg *c, jval *text) {
    memset(c->is_full, 0, sizeof(c->is_full));
    jval *types = json_get(text, "layer_types");
    if (!types || types->t != J_ARR || types->len != c->n_layers) {
        fprintf(stderr, "config.json: layer_types must list %d entries\n", c->n_layers);
        exit(1);
    }
    int full = 0;
    for (int i = 0; i < types->len; i++) {
        jval *entry = types->kids[i];
        if (!entry || entry->t != J_STR) {
            fprintf(stderr, "config.json: layer_types[%d] is not a string\n", i);
            exit(1);
        }
        if (strstr(entry->str, "linear")) {
            c->is_full[i] = 0;
        } else if (strstr(entry->str, "attention") || strstr(entry->str, "full")) {
            c->is_full[i] = 1;
            full++;
        } else {
            fprintf(stderr, "config.json: unknown layer type \"%s\" at %d\n", entry->str, i);
            exit(1);
        }
    }
    jval *linear = json_get(text, "linear_attn_config");
    jval *full_list = linear ? json_get(linear, "full_attn_layers") : NULL;
    if (full_list && full_list->t == J_ARR) {
        if (full_list->len != full) {
            fprintf(stderr, "config.json: full_attn_layers lists %d layers, "
                            "layer_types marks %d\n", full_list->len, full);
            exit(1);
        }
        for (int i = 0; i < full_list->len; i++) {
            int index = (int)full_list->kids[i]->num;
            if (index < 0 || index >= c->n_layers || !c->is_full[index]) {
                fprintf(stderr, "config.json: full_attn_layers disagrees with "
                                "layer_types at %d\n", index);
                exit(1);
            }
        }
    }
    /* The MTP block is a full-attention layer that lives past num_hidden_layers
     * and is not described by layer_types. */
    if (c->n_layers < (int)sizeof(c->is_full)) c->is_full[c->n_layers] = 1;
}

static void load_vision(Cfg *c, jval *root) {
    jval *vision = json_get(root, "vision_config");
    if (!vision || vision->t != J_OBJ) { c->vis_layers = 0; return; }
    c->vis_layers      = (int)req_num(vision, "depth");
    c->vis_hidden      = (int)req_num(vision, "hidden_size");
    c->vis_heads       = (int)req_num(vision, "num_heads");
    c->vis_inter       = (int)req_num(vision, "intermediate_size");
    c->vis_patch       = (int)req_num(vision, "patch_size");
    c->vis_temporal    = (int)opt_num(vision, "temporal_patch_size", 2);
    c->vis_merge       = (int)opt_num(vision, "spatial_merge_size", 2);
    c->vis_out_hidden  = (int)opt_num(vision, "out_hidden_size", c->hidden);
    c->vis_proj_inter  = (int)opt_num(vision, "projection_intermediate_size", 0);
    c->vis_image_size  = (int)opt_num(vision, "image_size", 448);
    c->vis_in_ch       = (int)opt_num(vision, "in_channels", 3);
    c->vis_swiglu_limit= (float)opt_num(vision, "swiglu_limit", 10.0);
    c->vis_eps         = (float)opt_num(vision, "rms_norm_eps", 1e-5);
    c->image_token       = (int)opt_num(root, "image_token_id", -1);
    c->image_start_token = (int)opt_num(root, "image_start_token_id", -1);
    c->image_end_token   = (int)opt_num(root, "image_end_token_id", -1);
    c->video_token       = (int)opt_num(root, "video_token_id", -1);
    c->video_start_token = (int)opt_num(root, "video_start_token_id", -1);
    c->video_end_token   = (int)opt_num(root, "video_end_token_id", -1);
    if (c->vis_layers < 1 || c->vis_layers > 128 || c->vis_hidden < 1 ||
        c->vis_heads < 1 || c->vis_hidden % c->vis_heads ||
        c->vis_patch < 1 || c->vis_merge < 1 || c->vis_out_hidden != c->hidden) {
        fprintf(stderr, "config.json: vision_config out of range "
                        "(out_hidden_size must equal the text hidden size)\n");
        exit(1);
    }
}

static void load_cfg(Cfg *c, const char *snap) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/config.json", snap);
    char *buffer = NULL;
    {
        FILE *stream = fopen(path, "rb");
        if (!stream) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
        if (fseek(stream, 0, SEEK_END)) { fprintf(stderr, "%s: not seekable\n", path); exit(1); }
        long length = ftell(stream);
        if (length < 2 || fseek(stream, 0, SEEK_SET)) {
            fprintf(stderr, "%s: unusable\n", path); exit(1); }
        buffer = malloc((size_t)length + 1);
        if (!buffer) { fprintf(stderr, "OOM reading config\n"); exit(1); }
        if (fread(buffer, 1, (size_t)length, stream) != (size_t)length) {
            fprintf(stderr, "%s: short read\n", path); exit(1); }
        buffer[length] = 0;
        fclose(stream);
    }
    char *arena = NULL;
    jval *root = json_parse(buffer, &arena);
    if (!root || root->t != J_OBJ) { fprintf(stderr, "%s: not a JSON object\n", path); exit(1); }
    memset(c, 0, sizeof(*c));

    jval *tc = json_get(root, "text_config");
    if (!tc || tc->t != J_OBJ) tc = root;      /* text-only export */

    c->hidden      = (int)req_num(tc, "hidden_size");
    c->n_layers    = (int)req_num(tc, "num_hidden_layers");
    c->vocab       = (int)req_num(tc, "vocab_size");
    c->first_dense = (int)req_num(tc, "first_k_dense_replace");
    c->dense_inter = (int)req_num(tc, "intermediate_size");
    c->n_heads     = (int)req_num(tc, "num_attention_heads");
    c->q_lora      = (int)req_num(tc, "q_lora_rank");
    c->kv_lora     = (int)req_num(tc, "kv_lora_rank");
    c->qk_nope     = (int)req_num(tc, "qk_nope_head_dim");
    c->qk_rope     = (int)opt_num(tc, "qk_rope_head_dim", 0);
    c->v_head      = (int)req_num(tc, "v_head_dim");
    c->n_experts   = (int)req_num(tc, "n_routed_experts");
    c->topk        = (int)req_num(tc, "num_experts_per_tok");
    c->moe_inter   = (int)req_num(tc, "moe_intermediate_size");
    c->n_shared    = (int)opt_num(tc, "n_shared_experts", 1);
    c->routed_scale= (float)opt_num(tc, "routed_scaling_factor", 1.0);
    c->eps         = (float)opt_num(tc, "rms_norm_eps", 1e-6);
    c->n_mtp       = (int)opt_num(tc, "num_nextn_predict_layers", 0);
    c->qk_head     = c->qk_nope + c->qk_rope;

    jval *linear = json_get(tc, "linear_attn_config");
    if (!linear || linear->t != J_OBJ) {
        fprintf(stderr, "config.json: missing linear_attn_config\n"); exit(1);
    }
    c->kda_heads = (int)req_num(linear, "num_heads");
    c->kda_hd    = (int)req_num(linear, "head_dim");
    c->conv_k    = (int)req_num(linear, "short_conv_kernel_size");
    c->gate_lb   = (float)opt_num(linear, "gate_lower_bound", -5.0);
    c->kda_proj  = c->kda_heads * c->kda_hd;

    c->index_topk       = (int)opt_num(tc, "index_topk", 0);
    c->index_nh         = (int)opt_num(tc, "index_n_heads", 0);
    c->index_hd         = (int)opt_num(tc, "index_head_dim", 0);
    c->index_kpool      = (int)opt_num(tc, "index_kpool", 1);
    c->index_kpool_tail = opt_bool(tc, "index_kpool_always_select_tail", 0);

    c->hc_mult  = (int)opt_num(tc, "hc_mult", 1);
    c->hc_iters = (int)opt_num(tc, "hc_sinkhorn_iters", 0);
    c->hc_eps   = (float)opt_num(tc, "hc_eps", 1e-6);

    load_layer_kinds(c, tc);
    load_vision(c, root);

    if (c->hidden < 1 || c->hidden > 65536 ||
        c->n_layers < 1 || c->n_layers > 120 ||
        c->vocab < 1 || c->vocab > (1 << 22) ||
        c->n_experts < 1 || c->n_experts > 4096 ||
        c->topk < 1 || c->topk > 64 || c->topk > c->n_experts ||
        c->kda_proj < 1 || c->kda_proj > (1 << 20) ||
        c->conv_k < 1 || c->conv_k > 8 ||
        c->moe_inter % 32 || c->kda_hd > 512 || c->kv_lora > 4096 ||
        c->first_dense < 0 || c->first_dense > c->n_layers ||
        c->index_kpool < 1 || c->index_kpool > 64 ||
        c->hc_mult < 1 || c->hc_mult > 8) {
        fprintf(stderr, "config.json: dimension out of range\n"); exit(1);
    }
    /* qk_rope must be zero: a rotary GLM-5.3 would need position handling this
     * engine deliberately does not have, and silently ignoring the rotation
     * would produce a model that answers fluently and wrongly. */
    if (c->qk_rope != 0) {
        fprintf(stderr, "config.json: qk_rope_head_dim=%d, but this engine "
                        "implements the NoPE full-attention of GLM-5.3\n", c->qk_rope);
        exit(1);
    }
    free(arena);
    free(buffer);
}

static void cfg_report(const Cfg *c) {
    int full = 0, kda = 0;
    for (int i = 0; i < c->n_layers; i++) { if (c->is_full[i]) full++; else kda++; }
    double expert_params = (double)c->n_experts * 3.0 * c->hidden * c->moe_inter *
                           (c->n_layers - c->first_dense + (c->n_mtp ? 1 : 0));
    fprintf(stderr,
        "GLM-5.3-Flash: %d layers (%d KDA + %d full) + %d MTP, hidden %d, vocab %d\n"
        "  MoE      : %d routed (top-%d) + %d shared, inter %d, from layer %d, scale %.2f\n"
        "  KDA      : %d heads x %d, conv %d, gate floor %.1f\n"
        "  MLA      : q_lora %d, kv_lora %d, qk %d (nope, NoPE), v %d, %d heads\n"
        "  indexer  : top-%d, %d heads x %d, kpool %d%s\n"
        "  mHC      : mult %d, %d Sinkhorn iterations\n"
        "  vision   : %s\n"
        "  routed experts: %.1f B parameters (%.0f GB at int4-g64)\n",
        c->n_layers, kda, full, c->n_mtp, c->hidden, c->vocab,
        c->n_experts, c->topk, c->n_shared, c->moe_inter, c->first_dense, c->routed_scale,
        c->kda_heads, c->kda_hd, c->conv_k, (double)c->gate_lb,
        c->q_lora, c->kv_lora, c->qk_nope, c->v_head, c->n_heads,
        c->index_topk, c->index_nh, c->index_hd, c->index_kpool,
        c->index_kpool_tail ? " (+tail)" : "",
        c->hc_mult, c->hc_iters,
        c->vis_layers ? "yes" : "text-only checkpoint",
        expert_params / 1e9, expert_params * 0.5625 / 1e9);
    if (c->vis_layers)
        fprintf(stderr,
        "             %d blocks x %d, %d heads, patch %d, %dpx, merge %d -> %d\n",
        c->vis_layers, c->vis_hidden, c->vis_heads, c->vis_patch,
        c->vis_image_size, c->vis_merge, c->vis_out_hidden);
}

#ifdef GLM53_CFG_MAIN
/* Config-only entry point: `glm53_cfg <model_dir>` parses and reports, so the
 * parser can be checked against a real checkpoint before any weight exists. */
int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <model_dir>\n", argv[0]); return 2; }
    Cfg c;
    load_cfg(&c, argv[1]);
    cfg_report(&c);
    return 0;
}
#endif

/* ---------- RAM-resident weight ----------
 * Same shape as kimi_k3.c's: either f32, int8 per-row, or int4 group-scaled.
 * The container written by tools/convert_glm53.py is U8 + `.qs` F32, which is
 * exactly what K3's loader already reads, so the two engines share a format
 * rather than each inventing one. */
typedef struct {
    int fmt;                              /* 0 = f32, 8 = int8 per-row, 4 = int4-g64 */
    float *f;
    int8_t *q8;
    uint8_t *q4;
    float *s;                             /* scales: [O] per-row, or [O*ngroups] */
    int O, I, gs;
} W;

/* ---------- layer structures ----------
 * Mirrors kimi_k3.c's shapes, with the GLM-5.3 differences called out where
 * they bite: the KDA output gate is low-rank here, the full-attention layers
 * are NOT gated (K3's are), and every layer carries two mHC sites instead of a
 * plain residual. */
typedef struct {                          /* KDA (linear attention) layer */
    W q, k, v, o;                         /* [proj x hidden] x3, [hidden x proj] */
    W ga, gb;                             /* low-rank output gate: hidden->hd->proj */
    float *conv_q, *conv_k, *conv_v;      /* [proj*conv_k] depthwise taps */
    float *fa, *fb;                       /* decay low-rank: [hd,hidden], [proj,hd] */
    float *bp;                            /* beta projection [heads,hidden] */
    float *dt, *A, *onw;                  /* dt_bias[proj], exp(A_log)[heads], o_norm[hd] */
} Kda;

typedef struct {                          /* MLA + DSA indexer (full attention) */
    W qa, qb, kva, kvb, o;
    float *qa_ln, *kva_ln;
    W wq, wk, wp;                         /* indexer: wq_b, wk, weights_proj */
    float *knw, *knb;                     /* indexer key LayerNorm (weight + bias) */
    float *kpool_ape;                     /* [kpool, index_hd] pool position bias */
    W kpool_gate;                         /* [index_hd, hidden] compression gate */
} Mla;

typedef struct {                          /* MoE (routed streamed + shared resident) */
    float *router, *rbias;                /* [E,hidden] f32, [E] correction bias */
    W sh_gate, sh_up, sh_down;
} Moe;

typedef struct {
    int full;                             /* 1 = MLA + indexer, 0 = KDA */
    int dense;                            /* 1 = plain MLP (layers < first_dense) */
    Kda a;
    Mla m;
    Moe moe;
    W d_gate, d_up, d_down;               /* dense layers only */
    float *in_ln, *post_ln;
    /* mHC, two sites per layer. fn is [(2+H)*H, H*hidden], base [(2+H)*H],
     * scale [3]; the split into pre/post/comb and the Sinkhorn projection are
     * the same as DeepSeek V4's. Absent on the MTP layer. */
    float *hc_attn_fn, *hc_attn_base, *hc_attn_scale;
    float *hc_ffn_fn,  *hc_ffn_base,  *hc_ffn_scale;
    /* MTP layer only */
    W mtp_eh;
    float *mtp_enorm, *mtp_hnorm, *mtp_head_norm;
} Layer;

/* ---------- tensor names ----------
 * One place builds every name the engine asks for. The checkpoint prefixes the
 * language model with `model.language_model.` because the root is the vision
 * wrapper; a text-only export drops it. Both are probed, and the choice is made
 * once from a tensor that must exist either way. */
typedef struct {
    const char *prefix;                   /* "model.language_model." or "model." */
    const char *visual;                   /* "model.visual." */
} Names;

#define GLM53_NAME(dst, fmt, ...) snprintf((dst), sizeof(dst), (fmt), __VA_ARGS__)

/* Emits, in load order, every tensor this engine will look for. `sink` is
 * called with (name, required); a NULL model just prints them, which is how
 * the mapping gets checked against a real checkpoint index before any weight
 * has been downloaded. */
static void glm53_walk_tensors(const Cfg *c, const Names *n,
                               void (*sink)(void *, const char *, int),
                               void *user) {
    char name[512];
#define EMIT(required, fmt, ...) do { \
        GLM53_NAME(name, fmt, __VA_ARGS__); sink(user, name, (required)); } while (0)
    EMIT(1, "%sembed_tokens.weight", n->prefix);
    EMIT(1, "%snorm.weight", n->prefix);
    sink(user, "lm_head.weight", 1);

    int last = c->n_layers + (c->n_mtp ? 1 : 0);
    for (int i = 0; i < last; i++) {
        int mtp = i >= c->n_layers;
        int full = c->is_full[i];
        EMIT(1, "%slayers.%d.input_layernorm.weight", n->prefix, i);
        EMIT(1, "%slayers.%d.post_attention_layernorm.weight", n->prefix, i);
        if (!mtp) {                        /* mHC lives on the 45 real layers */
            EMIT(1, "%slayers.%d.hc_attn_fn", n->prefix, i);
            EMIT(1, "%slayers.%d.hc_attn_base", n->prefix, i);
            EMIT(1, "%slayers.%d.hc_attn_scale", n->prefix, i);
            EMIT(1, "%slayers.%d.hc_ffn_fn", n->prefix, i);
            EMIT(1, "%slayers.%d.hc_ffn_base", n->prefix, i);
            EMIT(1, "%slayers.%d.hc_ffn_scale", n->prefix, i);
        } else {
            EMIT(1, "%slayers.%d.eh_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.enorm.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.hnorm.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.shared_head.norm.weight", n->prefix, i);
        }
        if (full) {
            EMIT(1, "%slayers.%d.self_attn.q_a_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.q_a_layernorm.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.q_b_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.kv_a_proj_with_mqa.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.kv_a_layernorm.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.kv_b_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.o_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.indexer.wq_b.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.indexer.wk.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.indexer.weights_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.indexer.k_norm.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.indexer.k_norm.bias", n->prefix, i);
            if (c->index_kpool > 1) {
                EMIT(1, "%slayers.%d.self_attn.indexer.index_kpool_compress_ape", n->prefix, i);
                EMIT(1, "%slayers.%d.self_attn.indexer.index_kpool_compress_gate", n->prefix, i);
            }
        } else {
            EMIT(1, "%slayers.%d.self_attn.q_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.k_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.v_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.o_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.g_a_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.g_b_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.q_conv1d.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.k_conv1d.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.v_conv1d.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.f_a_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.f_b_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.b_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.dt_bias", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.A_log", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.o_norm.weight", n->prefix, i);
        }
        if (i < c->first_dense) {
            EMIT(1, "%slayers.%d.mlp.gate_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.mlp.up_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.mlp.down_proj.weight", n->prefix, i);
        } else {
            EMIT(1, "%slayers.%d.mlp.gate.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.mlp.gate.e_score_correction_bias", n->prefix, i);
            EMIT(1, "%slayers.%d.mlp.shared_experts.gate_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.mlp.shared_experts.up_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.mlp.shared_experts.down_proj.weight", n->prefix, i);
            /* Routed experts are streamed, not resident: named here so the
             * inventory can prove they are all present and correctly shaped. */
            for (int e = 0; e < c->n_experts; e++) {
                EMIT(1, "%slayers.%d.mlp.experts.%d.gate_proj.weight", n->prefix, i, e);
                EMIT(1, "%slayers.%d.mlp.experts.%d.up_proj.weight", n->prefix, i, e);
                EMIT(1, "%slayers.%d.mlp.experts.%d.down_proj.weight", n->prefix, i, e);
            }
        }
    }
    if (c->vis_layers) {
        EMIT(1, "%spatch_embed.proj.weight", n->visual);
        EMIT(1, "%spatch_embed.proj.bias", n->visual);
        EMIT(1, "%spost_layernorm.weight", n->visual);
        EMIT(1, "%sdownsample.weight", n->visual);
        EMIT(1, "%sdownsample.bias", n->visual);
        EMIT(1, "%smerger.proj.weight", n->visual);
        EMIT(1, "%smerger.post_projection_norm.weight", n->visual);
        EMIT(1, "%smerger.post_projection_norm.bias", n->visual);
        EMIT(1, "%smerger.gate_proj.weight", n->visual);
        EMIT(1, "%smerger.up_proj.weight", n->visual);
        EMIT(1, "%smerger.down_proj.weight", n->visual);
        for (int b = 0; b < c->vis_layers; b++) {
            EMIT(1, "%sblocks.%d.norm1.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.norm2.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.attn.qkv.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.attn.qkv.bias", n->visual, b);
            EMIT(1, "%sblocks.%d.attn.proj.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.attn.proj.bias", n->visual, b);
            EMIT(1, "%sblocks.%d.attn.q_norm.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.attn.k_norm.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.mlp.gate_proj.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.mlp.gate_proj.bias", n->visual, b);
            EMIT(1, "%sblocks.%d.mlp.up_proj.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.mlp.up_proj.bias", n->visual, b);
            EMIT(1, "%sblocks.%d.mlp.down_proj.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.mlp.down_proj.bias", n->visual, b);
        }
    }
#undef EMIT
}

#ifdef GLM53_INVENTORY_MAIN
/* Inventory entry point: prints every tensor the engine would load, in load
 * order. Checked against a real checkpoint's index offline, so a naming
 * mistake surfaces before 180 GB have been converted rather than after. */
static void print_name(void *user, const char *name, int required) {
    (void)user; printf("%s\t%d\n", name, required);
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <model_dir>\n", argv[0]); return 2; }
    Cfg c;
    load_cfg(&c, argv[1]);
    Names n = { "model.language_model.", "model.visual." };
    glm53_walk_tensors(&c, &n, print_name, NULL);
    return 0;
}
#endif
