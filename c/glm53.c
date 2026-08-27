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
#include <stdarg.h>

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
    float routed_scale, eps, swiglu_limit;
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
    /* Il checkpoint reale dichiara first_k_dense_replace; la fixture tiny usa
     * mlp_layer_types. Accettiamo entrambi invece di imporne uno: sono la
     * stessa informazione detta in due modi, e rifiutare la fixture
     * significherebbe non poter provare il motore senza 194 GB. */
    {
        jval *fkd = json_get(tc, "first_k_dense_replace");
        if (fkd && fkd->t == J_NUM) {
            c->first_dense = (int)fkd->num;
        } else {
            jval *kinds = json_get(tc, "mlp_layer_types");
            if (!kinds || kinds->t != J_ARR) {
                fprintf(stderr, "config.json: serve first_k_dense_replace "
                                "oppure mlp_layer_types\n");
                exit(1);
            }
            c->first_dense = kinds->len;
            for (int i = 0; i < kinds->len; i++)
                if (kinds->kids[i]->t == J_STR && strstr(kinds->kids[i]->str, "sparse")) {
                    c->first_dense = i;
                    break;
                }
        }
    }
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
    c->swiglu_limit= (float)opt_num(tc, "swiglu_limit", 0.0);
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

#ifdef GLM53_CFG_MAIN_UNUSED
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

#ifdef GLM53_INVENTORY_MAIN_UNUSED
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

/* ================= forward =================
 * Il modello caricato in RAM come f32. Lo streaming degli esperti e la
 * quantizzazione a load time arrivano col modello vero; qui la priorita' e'
 * che i token siano quelli giusti, provati contro l'oracolo. */
#include "delta_attention.h"
#include "sparse_index.h"

typedef struct { const float *w; int rows, columns; } Mat;

typedef struct {
    /* comune */
    const float *in_ln, *post_ln;
    const float *hc_attn_fn, *hc_attn_base, *hc_attn_scale;
    const float *hc_ffn_fn,  *hc_ffn_base,  *hc_ffn_scale;
    /* KDA */
    Mat kq, kk, kv, ko, kga, kgb, kfa, kfb, kb;
    const float *conv, *dt, *alog, *onorm;
    /* MLA + indexer */
    Mat qa, qb, kva, kvb, o, iwq, iwk, iwp, ikpg;
    const float *qa_ln, *kva_ln, *ik_nw, *ik_nb, *ikpa;
    /* FFN */
    Mat dg, du, dd;                       /* denso */
    Mat rg, ru, rd;                       /* router / shared: gate,up,down */
    const float *router, *rbias;
    Mat *eg, *eu, *ed;                    /* esperti routed */
} GLayer;

typedef struct {
    Cfg c;
    shards S;
    const float *embed, *final_norm, *head;
    GLayer *layer;
    char prefix[64];
} GModel;

static const float *load_f32(GModel *m, const char *fmt, ...) {
    char name[512];
    va_list args; va_start(args, fmt); vsnprintf(name, sizeof(name), fmt, args); va_end(args);
    st_tensor *t = st_find(&m->S, name);
    if (!t) { fprintf(stderr, "manca il tensore %s\n", name); exit(1); }
    float *buffer = malloc((size_t)t->numel * sizeof(float));
    if (!buffer) { fprintf(stderr, "OOM su %s\n", name); exit(1); }
    st_read_f32_cap(&m->S, name, buffer, t->numel, 0);
    return buffer;
}

static Mat load_mat(GModel *m, const char *fmt, ...) {
    char name[512];
    va_list args; va_start(args, fmt); vsnprintf(name, sizeof(name), fmt, args); va_end(args);
    st_tensor *t = st_find(&m->S, name);
    if (!t) { fprintf(stderr, "manca la matrice %s\n", name); exit(1); }
    if (t->rank != 2) { fprintf(stderr, "%s: rank %d, attesa 2\n", name, t->rank); exit(1); }
    float *buffer = malloc((size_t)t->numel * sizeof(float));
    st_read_f32_cap(&m->S, name, buffer, t->numel, 0);
    Mat mat = { buffer, (int)t->shape[0], (int)t->shape[1] };
    return mat;
}

/* out[rows] = W x, con W in layout [rows, columns] come transformers. */
static void mv(float *out, const Mat *w, const float *x) {
    for (int r = 0; r < w->rows; r++) {
        const float *row = w->w + (size_t)r * w->columns;
        float sum = 0.0f;
        for (int c = 0; c < w->columns; c++) sum += row[c] * x[c];
        out[r] = sum;
    }
}

static void rms(float *out, const float *x, const float *w, int n, float eps) {
    float square = 0.0f;
    for (int i = 0; i < n; i++) square += x[i] * x[i];
    float inverse = 1.0f / sqrtf(square / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inverse * w[i];
}

static void layer_norm(float *out, const float *x, const float *w, const float *b,
                       int n, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    float variance = 0.0f;
    for (int i = 0; i < n; i++) variance += (x[i] - mean) * (x[i] - mean);
    variance /= n;
    float inverse = 1.0f / sqrtf(variance + eps);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inverse * w[i] + (b ? b[i] : 0.0f);
}

static float sigmoidf_(float x) {
    return x >= 0.0f ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x));
}
static float siluf_(float x) { return x / (1.0f + expf(-x)); }

/* SwiGLU clampata: il gate ha solo il tetto, up e' limitato da entrambi i lati.
 * Vale sia per l'MLP denso che per gli esperti -- il testo di GLM-5.3 NON usa
 * la SiLU semplice, ed e' un errore che darebbe un modello che parla bene e
 * sbaglia. */
static void swiglu_clamped(float *gate, const float *up, int n, float limit) {
    for (int i = 0; i < n; i++) {
        float g = gate[i] > limit ? limit : gate[i];
        float u = up[i] < -limit ? -limit : (up[i] > limit ? limit : up[i]);
        gate[i] = siluf_(g) * u;
    }
}

static void mlp3(float *out, const float *x, const Mat *g, const Mat *u, const Mat *d,
                 float limit, float *sg, float *su) {
    mv(sg, g, x); mv(su, u, x);
    swiglu_clamped(sg, su, g->rows, limit);
    mv(out, d, sg);
}

/* ---------- KDA: proiezioni, gate, ricorrenza ---------- */
static void kda_layer(const Cfg *c, const GLayer *l, const float *x, int tokens,
                      float *out, float *state, float *window, float *scratch) {
    const int P = c->kda_proj, H = c->kda_heads, D = c->kda_hd;
    float *qkv = malloc((size_t)3 * P * sizeof(float));
    float *gate = malloc((size_t)P * sizeof(float));
    float *decay = malloc((size_t)P * sizeof(float));
    float *beta = malloc((size_t)H * sizeof(float));
    float *low = malloc((size_t)D * sizeof(float));
    float *core = malloc((size_t)P * sizeof(float));
    for (int t = 0; t < tokens; t++) {
        const float *row = x + (size_t)t * c->hidden;
        mv(qkv, &l->kq, row);
        mv(qkv + P, &l->kk, row);
        mv(qkv + 2 * P, &l->kv, row);
        /* decadimento: gate_lower_bound * sigmoid(exp(A_log[h]) * (W_fb W_fa x + dt_bias)) */
        mv(low, &l->kfa, row);
        mv(decay, &l->kfb, low);
        for (int h = 0; h < H; h++)
            for (int d = 0; d < D; d++) {
                int i = h * D + d;
                decay[i] = c->gate_lb * sigmoidf_(expf(l->alog[h]) * (decay[i] + l->dt[i]));
            }
        mv(beta, &l->kb, row);
        for (int h = 0; h < H; h++) beta[h] = sigmoidf_(beta[h]);
        coli_kda_step(core, state, window, qkv, l->conv, decay, beta,
                      H, D, D, c->conv_k, 1e-6f, scratch);
        /* uscita: RMSNorm per testa, pesata da o_norm, moltiplicata dal gate
         * low-rank, poi la proiezione di uscita. */
        mv(low, &l->kga, row);
        mv(gate, &l->kgb, low);
        float *normed = qkv;                         /* riuso: 3P >= P */
        for (int h = 0; h < H; h++) {
            const float *src = core + (size_t)h * D;
            float *dst = normed + (size_t)h * D;
            float square = 0.0f;
            for (int d = 0; d < D; d++) square += src[d] * src[d];
            float inverse = 1.0f / sqrtf(square / D + c->eps);
            for (int d = 0; d < D; d++)
                dst[d] = src[d] * inverse * l->onorm[d] * sigmoidf_(gate[(size_t)h * D + d]);
        }
        mv(out + (size_t)t * c->hidden, &l->ko, normed);
    }
    free(core); free(low); free(beta); free(decay); free(gate); free(qkv);
}

/* ---------- MLA + indexer con k-pool ---------- */
static void mla_layer(const Cfg *c, const GLayer *l, const float *x, int tokens, float *out) {
    const int H = c->n_heads, QK = c->qk_nope, V = c->v_head;
    const int IH = c->index_nh, ID = c->index_hd;
    float *qa = malloc((size_t)tokens * c->q_lora * sizeof(float));
    float *queries = malloc((size_t)tokens * H * QK * sizeof(float));
    float *keys = malloc((size_t)tokens * H * QK * sizeof(float));
    float *values = malloc((size_t)tokens * H * V * sizeof(float));
    float *latent = malloc((size_t)c->kv_lora * sizeof(float));
    float *expanded = malloc((size_t)H * (QK + V) * sizeof(float));
    float *iq = malloc((size_t)tokens * IH * ID * sizeof(float));
    float *ik = malloc((size_t)tokens * ID * sizeof(float));
    float *gates = malloc((size_t)tokens * ID * sizeof(float));
    float *head_w = malloc((size_t)tokens * IH * sizeof(float));
    unsigned char *valid = malloc((size_t)tokens);
    memset(valid, 1, (size_t)tokens);

    for (int t = 0; t < tokens; t++) {
        const float *row = x + (size_t)t * c->hidden;
        float *qn = qa + (size_t)t * c->q_lora;
        mv(qn, &l->qa, row);
        rms(qn, qn, l->qa_ln, c->q_lora, c->eps);
        mv(queries + (size_t)t * H * QK, &l->qb, qn);
        mv(latent, &l->kva, row);
        rms(latent, latent, l->kva_ln, c->kv_lora, c->eps);
        mv(expanded, &l->kvb, latent);
        for (int h = 0; h < H; h++) {
            const float *src = expanded + (size_t)h * (QK + V);
            memcpy(keys + ((size_t)t * H + h) * QK, src, (size_t)QK * sizeof(float));
            memcpy(values + ((size_t)t * H + h) * V, src + QK, (size_t)V * sizeof(float));
        }
        /* indexer: le query vengono dal q_a normalizzato, le chiavi dall'hidden
         * con LayerNorm (con bias), e i pesi per testa sono scalati da IH^-0.5 */
        mv(iq + (size_t)t * IH * ID, &l->iwq, qn);
        float *kraw = ik + (size_t)t * ID;
        mv(kraw, &l->iwk, row);
        layer_norm(kraw, kraw, l->ik_nw, l->ik_nb, ID, 1e-5f);
        mv(gates + (size_t)t * ID, &l->ikpg, row);
        mv(head_w + (size_t)t * IH, &l->iwp, row);
        for (int h = 0; h < IH; h++) head_w[(size_t)t * IH + h] /= sqrtf((float)IH);
    }

    const int width = coli_sparse_index_width(c->index_topk, c->index_kpool, c->index_kpool_tail);
    int *selected = malloc((size_t)tokens * width * sizeof(int));
    if (coli_sparse_index_select(selected, iq, ik, gates, head_w, l->ikpa, valid,
                                 tokens, IH, ID, c->index_kpool, c->index_topk,
                                 c->index_kpool_tail)) {
        fprintf(stderr, "selezione indexer fallita\n"); exit(1);
    }
    float *context = malloc((size_t)tokens * H * V * sizeof(float));
    if (coli_sparse_attention(context, queries, keys, values, selected,
                              tokens, width, H, QK, V)) {
        fprintf(stderr, "attenzione sparsa fallita\n"); exit(1);
    }
    for (int t = 0; t < tokens; t++)
        mv(out + (size_t)t * c->hidden, &l->o, context + (size_t)t * H * V);

    free(context); free(selected); free(valid); free(head_w); free(gates);
    free(ik); free(iq); free(expanded); free(latent);
    free(values); free(keys); free(queries); free(qa);
}

/* ---------- FFN: denso oppure MoE ---------- */
static void ffn_layer(const Cfg *c, const GLayer *l, int index, const float *x,
                      int tokens, float *out) {
    const int inter = index < c->first_dense ? c->dense_inter : c->moe_inter;
    float *sg = malloc((size_t)(c->dense_inter > c->moe_inter ? c->dense_inter : c->moe_inter)
                       * sizeof(float));
    float *su = malloc((size_t)(c->dense_inter > c->moe_inter ? c->dense_inter : c->moe_inter)
                       * sizeof(float));
    float *tmp = malloc((size_t)c->hidden * sizeof(float));
    float *score = malloc((size_t)c->n_experts * sizeof(float));
    (void)inter;
    for (int t = 0; t < tokens; t++) {
        const float *row = x + (size_t)t * c->hidden;
        float *dst = out + (size_t)t * c->hidden;
        if (index < c->first_dense) {
            mlp3(dst, row, &l->dg, &l->du, &l->dd, c->swiglu_limit, sg, su);
            continue;
        }
        /* esperto condiviso, sempre attivo */
        mlp3(dst, row, &l->rg, &l->ru, &l->rd, c->swiglu_limit, sg, su);
        /* router: sigmoide sui logit; la selezione usa score+bias, il PESO usa
         * lo score puro -- la distinzione e' sottile e sbagliarla cambia quali
         * esperti contano quanto. */
        for (int e = 0; e < c->n_experts; e++) {
            const float *w = l->router + (size_t)e * c->hidden;
            float sum = 0.0f;
            for (int d = 0; d < c->hidden; d++) sum += w[d] * row[d];
            score[e] = sigmoidf_(sum);
        }
        int chosen[64];
        float weight[64], total = 0.0f;
        for (int k = 0; k < c->topk; k++) {
            int best = -1; float value = -INFINITY;
            for (int e = 0; e < c->n_experts; e++) {
                int used = 0;
                for (int j = 0; j < k; j++) if (chosen[j] == e) { used = 1; break; }
                float choice = score[e] + (l->rbias ? l->rbias[e] : 0.0f);
                if (!used && choice > value) { value = choice; best = e; }
            }
            chosen[k] = best;
            weight[k] = score[best];
            total += weight[k];
        }
        for (int k = 0; k < c->topk; k++) {
            float scale = weight[k] / (total + 1e-20f) * c->routed_scale;
            mlp3(tmp, row, &l->eg[chosen[k]], &l->eu[chosen[k]], &l->ed[chosen[k]],
                 c->swiglu_limit, sg, su);
            for (int d = 0; d < c->hidden; d++) dst[d] += scale * tmp[d];
        }
    }
    free(score); free(tmp); free(su); free(sg);
}

/* ---------- caricamento ---------- */
static void model_load(GModel *m, const char *dir) {
    load_cfg(&m->c, dir);
    st_init(&m->S, dir);
    /* Il checkpoint reale annida il modello testuale sotto il wrapper vision;
     * un export solo-testo no. Si sceglie una volta, da un tensore che deve
     * esistere in entrambe le forme. */
    snprintf(m->prefix, sizeof(m->prefix), "model.language_model.");
    char probe[256];
    snprintf(probe, sizeof(probe), "%sembed_tokens.weight", m->prefix);
    if (!st_find(&m->S, probe)) snprintf(m->prefix, sizeof(m->prefix), "model.");
    const char *P = m->prefix;

    m->embed = load_f32(m, "%sembed_tokens.weight", P);
    m->final_norm = load_f32(m, "%snorm.weight", P);
    m->head = st_find(&m->S, "lm_head.weight") ? load_f32(m, "lm_head.weight") : m->embed;
    m->layer = calloc((size_t)m->c.n_layers, sizeof(*m->layer));

    for (int i = 0; i < m->c.n_layers; i++) {
        GLayer *l = &m->layer[i];
        l->in_ln = load_f32(m, "%slayers.%d.input_layernorm.weight", P, i);
        l->post_ln = load_f32(m, "%slayers.%d.post_attention_layernorm.weight", P, i);
        l->hc_attn_fn = load_f32(m, "%slayers.%d.hc_attn_fn", P, i);
        l->hc_attn_base = load_f32(m, "%slayers.%d.hc_attn_base", P, i);
        l->hc_attn_scale = load_f32(m, "%slayers.%d.hc_attn_scale", P, i);
        l->hc_ffn_fn = load_f32(m, "%slayers.%d.hc_ffn_fn", P, i);
        l->hc_ffn_base = load_f32(m, "%slayers.%d.hc_ffn_base", P, i);
        l->hc_ffn_scale = load_f32(m, "%slayers.%d.hc_ffn_scale", P, i);
        if (m->c.is_full[i]) {
            l->qa = load_mat(m, "%slayers.%d.self_attn.q_a_proj.weight", P, i);
            l->qa_ln = load_f32(m, "%slayers.%d.self_attn.q_a_layernorm.weight", P, i);
            l->qb = load_mat(m, "%slayers.%d.self_attn.q_b_proj.weight", P, i);
            l->kva = load_mat(m, "%slayers.%d.self_attn.kv_a_proj_with_mqa.weight", P, i);
            l->kva_ln = load_f32(m, "%slayers.%d.self_attn.kv_a_layernorm.weight", P, i);
            l->kvb = load_mat(m, "%slayers.%d.self_attn.kv_b_proj.weight", P, i);
            l->o = load_mat(m, "%slayers.%d.self_attn.o_proj.weight", P, i);
            l->iwq = load_mat(m, "%slayers.%d.self_attn.indexer.wq_b.weight", P, i);
            l->iwk = load_mat(m, "%slayers.%d.self_attn.indexer.wk.weight", P, i);
            l->iwp = load_mat(m, "%slayers.%d.self_attn.indexer.weights_proj.weight", P, i);
            l->ik_nw = load_f32(m, "%slayers.%d.self_attn.indexer.k_norm.weight", P, i);
            l->ik_nb = load_f32(m, "%slayers.%d.self_attn.indexer.k_norm.bias", P, i);
            if (m->c.index_kpool > 1) {
                l->ikpa = load_f32(m, "%slayers.%d.self_attn.indexer.index_kpool_compress_ape", P, i);
                l->ikpg = load_mat(m, "%slayers.%d.self_attn.indexer.index_kpool_compress_gate", P, i);
            }
        } else {
            l->kq = load_mat(m, "%slayers.%d.self_attn.q_proj.weight", P, i);
            l->kk = load_mat(m, "%slayers.%d.self_attn.k_proj.weight", P, i);
            l->kv = load_mat(m, "%slayers.%d.self_attn.v_proj.weight", P, i);
            l->ko = load_mat(m, "%slayers.%d.self_attn.o_proj.weight", P, i);
            l->kga = load_mat(m, "%slayers.%d.self_attn.g_a_proj.weight", P, i);
            l->kgb = load_mat(m, "%slayers.%d.self_attn.g_b_proj.weight", P, i);
            l->kfa = load_mat(m, "%slayers.%d.self_attn.f_a_proj.weight", P, i);
            l->kfb = load_mat(m, "%slayers.%d.self_attn.f_b_proj.weight", P, i);
            l->kb = load_mat(m, "%slayers.%d.self_attn.b_proj.weight", P, i);
            l->dt = load_f32(m, "%slayers.%d.self_attn.dt_bias", P, i);
            l->alog = load_f32(m, "%slayers.%d.self_attn.A_log", P, i);
            l->onorm = load_f32(m, "%slayers.%d.self_attn.o_norm.weight", P, i);
            /* Il checkpoint tiene q/k/v conv separate; la ricorrenza le vuole
             * concatenate nello stesso ordine di qkv. */
            {
                int width = m->c.kda_proj * m->c.conv_k;
                float *conv = malloc((size_t)3 * width * sizeof(float));
                const char *parts[3] = { "q_conv1d", "k_conv1d", "v_conv1d" };
                for (int p = 0; p < 3; p++) {
                    const float *piece = load_f32(m, "%slayers.%d.self_attn.%s.weight",
                                                  P, i, parts[p]);
                    memcpy(conv + (size_t)p * width, piece, (size_t)width * sizeof(float));
                    free((void *)piece);
                }
                l->conv = conv;
            }
        }
        if (i < m->c.first_dense) {
            l->dg = load_mat(m, "%slayers.%d.mlp.gate_proj.weight", P, i);
            l->du = load_mat(m, "%slayers.%d.mlp.up_proj.weight", P, i);
            l->dd = load_mat(m, "%slayers.%d.mlp.down_proj.weight", P, i);
        } else {
            l->router = load_f32(m, "%slayers.%d.mlp.gate.weight", P, i);
            l->rbias = st_find(&m->S, (snprintf(probe, sizeof(probe),
                        "%slayers.%d.mlp.gate.e_score_correction_bias", P, i), probe))
                       ? load_f32(m, "%s", probe) : NULL;
            l->rg = load_mat(m, "%slayers.%d.mlp.shared_experts.gate_proj.weight", P, i);
            l->ru = load_mat(m, "%slayers.%d.mlp.shared_experts.up_proj.weight", P, i);
            l->rd = load_mat(m, "%slayers.%d.mlp.shared_experts.down_proj.weight", P, i);
            l->eg = malloc((size_t)m->c.n_experts * sizeof(Mat));
            l->eu = malloc((size_t)m->c.n_experts * sizeof(Mat));
            l->ed = malloc((size_t)m->c.n_experts * sizeof(Mat));
            for (int e = 0; e < m->c.n_experts; e++) {
                l->eg[e] = load_mat(m, "%slayers.%d.mlp.experts.%d.gate_proj.weight", P, i, e);
                l->eu[e] = load_mat(m, "%slayers.%d.mlp.experts.%d.up_proj.weight", P, i, e);
                l->ed[e] = load_mat(m, "%slayers.%d.mlp.experts.%d.down_proj.weight", P, i, e);
            }
        }
    }
}

/* ---------- il passaggio completo ---------- */
static float *forward(GModel *m, const int *tokens, int n) {
    const Cfg *c = &m->c;
    const int H = c->hc_mult, D = c->hidden;
    float *streams = malloc((size_t)n * H * D * sizeof(float));
    float *next = malloc((size_t)n * H * D * sizeof(float));
    float *collapsed = malloc((size_t)n * D * sizeof(float));
    float *normed = malloc((size_t)n * D * sizeof(float));
    float *branch = malloc((size_t)n * D * sizeof(float));
    float *post = malloc((size_t)n * H * sizeof(float));
    float *comb = malloc((size_t)n * H * H * sizeof(float));
    /* l'embedding entra replicato in ognuno degli H flussi residui */
    for (int t = 0; t < n; t++)
        for (int h = 0; h < H; h++)
            memcpy(streams + ((size_t)t * H + h) * D,
                   m->embed + (size_t)tokens[t] * D, (size_t)D * sizeof(float));

    float *state = NULL, *window = NULL, *scratch = NULL;
    if (c->kda_proj) {
        state = malloc((size_t)c->kda_heads * c->kda_hd * c->kda_hd * sizeof(float));
        window = malloc((size_t)3 * c->kda_proj * c->conv_k * sizeof(float));
        scratch = malloc((size_t)coli_kda_scratch_floats(c->kda_heads, c->kda_hd, c->kda_hd)
                         * sizeof(float));
    }

    for (int i = 0; i < c->n_layers; i++) {
        GLayer *l = &m->layer[i];
        for (int site = 0; site < 2; site++) {
            const float *fn = site ? l->hc_ffn_fn : l->hc_attn_fn;
            const float *base = site ? l->hc_ffn_base : l->hc_attn_base;
            const float *scale = site ? l->hc_ffn_scale : l->hc_attn_scale;
            for (int t = 0; t < n; t++)
                coli_hc_pre(collapsed + (size_t)t * D, post + (size_t)t * H,
                            comb + (size_t)t * H * H, streams + (size_t)t * H * D,
                            fn, scale, base, H, D, c->hc_iters, c->eps, c->hc_eps);
            for (int t = 0; t < n; t++)
                rms(normed + (size_t)t * D, collapsed + (size_t)t * D,
                    site ? l->post_ln : l->in_ln, D, c->eps);
            if (!site) {
                if (c->is_full[i]) mla_layer(c, l, normed, n, branch);
                else {
                    memset(state, 0, (size_t)c->kda_heads * c->kda_hd * c->kda_hd * sizeof(float));
                    memset(window, 0, (size_t)3 * c->kda_proj * c->conv_k * sizeof(float));
                    kda_layer(c, l, normed, n, branch, state, window, scratch);
                }
            } else {
                ffn_layer(c, l, i, normed, n, branch);
            }
            for (int t = 0; t < n; t++)
                coli_hc_post(next + (size_t)t * H * D, branch + (size_t)t * D,
                             streams + (size_t)t * H * D, post + (size_t)t * H,
                             comb + (size_t)t * H * H, H, D);
            float *swap = streams; streams = next; next = swap;
        }
    }
    /* i flussi si richiudono con una media NON pesata */
    for (int t = 0; t < n; t++)
        for (int d = 0; d < D; d++) {
            float sum = 0.0f;
            for (int h = 0; h < H; h++) sum += streams[((size_t)t * H + h) * D + d];
            collapsed[(size_t)t * D + d] = sum / H;
        }
    for (int t = 0; t < n; t++)
        rms(normed + (size_t)t * D, collapsed + (size_t)t * D, m->final_norm, D, c->eps);

    float *logits = malloc((size_t)n * c->vocab * sizeof(float));
    Mat head = { m->head, c->vocab, D };
    for (int t = 0; t < n; t++) mv(logits + (size_t)t * c->vocab, &head, normed + (size_t)t * D);

    free(scratch); free(window); free(state);
    free(comb); free(post); free(branch); free(normed); free(collapsed);
    free(next); free(streams);
    return logits;
}

/* ---------- CLI ----------
 * Due modi, entrambi pensati per essere confrontati con ref.json:
 *   --ids a,b,c            teacher forcing: stampa l'argmax a ogni posizione
 *   --ids a,b,c --greedy N genera N token continuando dal prompt
 * Il tokenizzatore non serve: l'oracolo parla in id. */
static int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

int main(int argc, char **argv) {
    const char *dir = NULL, *ids = NULL;
    int greedy = 0, show_logits = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids = argv[++i];
        else if (!strcmp(argv[i], "--greedy") && i + 1 < argc) greedy = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--logits")) show_logits = 1;
        else { fprintf(stderr, "argomento sconosciuto: %s\n", argv[i]); return 2; }
    }
    if (!dir || !ids) {
        fprintf(stderr, "uso: %s --model DIR --ids a,b,c [--greedy N] [--logits]\n", argv[0]);
        return 2;
    }
    int capacity = 1024, count = 0;
    int *tokens = malloc((size_t)capacity * sizeof(int));
    for (const char *p = ids; *p; ) {
        char *end;
        long value = strtol(p, &end, 10);
        if (end == p) break;
        if (count == capacity) tokens = realloc(tokens, (size_t)(capacity *= 2) * sizeof(int));
        tokens[count++] = (int)value;
        p = (*end == ',') ? end + 1 : end;
    }
    if (!count) { fprintf(stderr, "nessun id nel prompt\n"); return 2; }

    GModel model;
    model_load(&model, dir);
    if (getenv("GLM53_VERBOSE")) cfg_report(&model.c);

    float *logits = forward(&model, tokens, count);
    printf("teacher_forcing");
    for (int t = 0; t < count; t++)
        printf(" %d", argmax(logits + (size_t)t * model.c.vocab, model.c.vocab));
    printf("\n");
    if (show_logits) {
        printf("last_logits");
        const float *last = logits + (size_t)(count - 1) * model.c.vocab;
        for (int v = 0; v < model.c.vocab; v++) printf(" %.9g", last[v]);
        printf("\n");
    }
    if (greedy > 0) {
        int total = count;
        int *sequence = malloc((size_t)(count + greedy) * sizeof(int));
        memcpy(sequence, tokens, (size_t)count * sizeof(int));
        printf("greedy");
        for (int step = 0; step < greedy; step++) {
            free(logits);
            logits = forward(&model, sequence, total);
            int next = argmax(logits + (size_t)(total - 1) * model.c.vocab, model.c.vocab);
            sequence[total++] = next;
            printf(" %d", next);
        }
        printf("\n");
        free(sequence);
    }
    free(logits);
    free(tokens);
    return 0;
}
