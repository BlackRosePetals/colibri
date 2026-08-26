/* Native Qwen3.8-Flash-Next text core.
 *
 * This header is included once by qwen38.c after its tokenizer and protocol
 * helpers.  It intentionally consumes the official HF safetensors layout:
 * the multimodal wrapper's `model.language_model` namespace and the standalone
 * text model's `model` namespace are both accepted.  Vision and MTP tensors are
 * indexed by st.h but are never read.
 */
#ifndef COLI_QWEN38_CORE_H
#define COLI_QWEN38_CORE_H

#define Q38_MAX_LAYERS 512
#define Q38_MAX_EXPERTS 1024
#define Q38_MAX_TOPK 256
#define Q38_MAX_PLE_PARTS 512

typedef struct {
    int hidden, layers, vocab, max_positions, eos_id;
    float eps, theta;
    int hc_count, hc_rank, hc_width;
    int q_heads, kv_heads, head_dim, rotary_dim;
    int idx_qheads, idx_kheads, idx_dim, idx_budget, idx_ratio;
    int experts, topk, inter, shared_inter, norm_topk;
    int dn_kheads, dn_vheads, dn_kdim, dn_vdim, dn_convk, dn_conv_dim;
    int ple_layer, ple_dim, ple_convk, ngram_size, heads_per_ngram;
    int ngram_heads, ngram_head_dim, ngram_parts;
    uint8_t *is_attn;
} Cfg;

typedef struct { float *norm, *down, *up, *inject; } GatedResidual;

typedef struct {
    GatedResidual attn_gr, mlp_gr;
    float *router, *sh_g, *sh_u, *sh_d, *sh_gate;
    float *q, *k, *v, *o, *qn, *kn;
    float *idx_qk, *idx_qn, *idx_kn;
    float *dn_qkv, *dn_z, *dn_b, *dn_a, *dn_conv;
    float *dn_dtbias, *dn_alog, *dn_norm, *dn_out;
    float *ple_key, *ple_value, *ple_norm_key, *ple_norm_query;
    float *ple_norm_conv, *ple_conv;
} Layer;

typedef struct { int eid; float *gate, *up, *down; uint64_t used; } Slot;
typedef struct { Slot *slots; int *by_expert, n, cap; } LCache;

typedef struct {
    Cfg c;
    shards S;
    char prefix[32];
    float *embed, *lm_head;
    GatedResidual final_gr;
    Layer *L;
    LCache *cache;
    uint64_t clock, hits, miss;
    float **DN_rec, **DN_conv;
    float **K, **V, **IK;
    int kv_len, kv_cap, max_t;
    st_tensor *ple_parts[Q38_MAX_PLE_PARTS];
    char ple_part_names[Q38_MAX_PLE_PARTS][320];
    int64_t ple_part_start[Q38_MAX_PLE_PARTS + 1];
    int ple_part_count;
    float ple_weight_scale;
    int64_t ple_multipliers[3], ple_head_vocab[64], ple_head_offset[64];
    int64_t *ple_history;
    float *PLE_conv_state;
    int ple_history_len;
    int range_begin, range_end;
    double dense_load_s;
} Model;

static float *g_last_logit;
static int g_capture_last_logit;

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

#if defined(__APPLE__)
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF,&r); return r.ru_maxrss/1073741824.0; }
#else
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF,&r); return r.ru_maxrss/1048576.0; }
#endif

static float *falloc(int64_t n) {
    if (n < 0 || (uint64_t)n > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "invalid float allocation: %lld\n", (long long)n); exit(1);
    }
    float *p = (float *)malloc((size_t)n * sizeof(float));
    if (!p && n) { fprintf(stderr, "OOM allocating %lld floats\n", (long long)n); exit(1); }
    return p;
}

/* W is row-major [O,I], y=x@W^T. */
static void q38_matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float a = 0.f;
            for (int i = 0; i < I; i++) a += xs[i] * w[i];
            y[(int64_t)s * O + o] = a;
        }
    }
}

static inline float q38_sigmoid(float x) {
    if (x >= 0.f) { float z=expf(-x); return 1.f/(1.f+z); }
    float z=expf(x); return z/(1.f+z);
}
static inline float q38_silu(float x) { return x * q38_sigmoid(x); }
static inline float q38_softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }

/* Qwen4-Exp RMSNorms are zero-centered: the learned scale is 1+weight. */
static void q38_rms0(float *out, const float *x, const float *w, int n, float eps) {
    double ss=0.0; for (int i=0;i<n;i++) ss+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ss/n)+eps);
    for (int i=0;i<n;i++) out[i]=x[i]*r*(1.f+w[i]);
}

/* DeltaNet's RMSNormGated is inherited from Qwen3-Next and is not zero-centered. */
static void q38_rmsg(float *out,const float *x,const float *gate,const float *w,
                     int n,float eps,int sigmoid_gate) {
    double ss=0.0; for(int i=0;i<n;i++) ss+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ss/n)+eps);
    for(int i=0;i<n;i++) out[i]=x[i]*r*w[i]*(sigmoid_gate?q38_sigmoid(gate[i]):q38_silu(gate[i]));
}

static void q38_rope(float *x, int dim, int rotary_dim, int pos, float theta) {
    int half=rotary_dim/2;
    for(int i=0;i<half;i++) {
        float ang=(float)pos/powf(theta,(float)(2*i)/rotary_dim);
        float co=cosf(ang), si=sinf(ang), a=x[i], b=x[i+half];
        x[i]=a*co-b*si; x[i+half]=b*co+a*si;
    }
    (void)dim;
}

static jval *q38_obj(jval *o,const char *key) {
    jval *v=json_get(o,key);
    if(v&&v->t!=J_OBJ){fprintf(stderr,"config.json: %s must be an object\n",key);exit(1);}
    return v;
}
static double q38_num(jval *o,const char *key,double def,int required) {
    jval *v=json_get(o,key);
    if(v&&v->t==J_NUM) return v->num;
    if(v){fprintf(stderr,"config.json: %s must be numeric\n",key);exit(1);}
    if(required){fprintf(stderr,"config.json: missing numeric %s\n",key);exit(1);} return def;
}
/* JSON numbers are doubles in the small parser.  Never narrow an unchecked
 * value to an int: besides accepting fractional dimensions, a huge finite
 * value can become implementation-defined before q38_validate_cfg sees it. */
static int q38_num_int(jval *o,const char *key,double def,int required,
                       int min_value,int max_value) {
    double value=q38_num(o,key,def,required);
    if(!isfinite(value)||floor(value)!=value||value<(double)min_value||
       value>(double)max_value){
        fprintf(stderr,"config.json: %s must be an integer in [%d,%d]\n",
                key,min_value,max_value);exit(1);
    }
    return (int)value;
}
static int q38_derived_product(const char *name,int left,int right) {
    if(left<0||right<0||(uint64_t)left*(uint64_t)right>(uint64_t)INT_MAX){
        fprintf(stderr,"[qwen38 config] derived %s overflows int\n",name);exit(1);
    }
    return left*right;
}
static int q38_derived_sum(const char *name,int left,int right) {
    if(left<0||right<0||left>INT_MAX-right){
        fprintf(stderr,"[qwen38 config] derived %s overflows int\n",name);exit(1);
    }
    return left+right;
}
static int q38_bool(jval *o,const char *key,int def) {
    jval *v=json_get(o,key);
    if(v&&v->t!=J_BOOL){fprintf(stderr,"config.json: %s must be boolean\n",key);exit(1);}
    return v?v->boolean:def;
}
static const char *q38_string(jval *o,const char *key,const char *def) {
    jval *v=json_get(o,key);
    if(v&&v->t!=J_STR){fprintf(stderr,"config.json: %s must be a string\n",key);exit(1);}
    return v?v->str:def;
}
static void q38_require_string(jval *o,const char *key,const char *expected) {
    const char *value=q38_string(o,key,expected);
    if(strcmp(value,expected)){
        fprintf(stderr,"config.json: unsupported %s=%s (expected %s)\n",
                key,value,expected);exit(1);
    }
}
static void q38_require_present_string(jval *o,const char *key,
                                        const char *expected) {
    jval *value=json_get(o,key);
    if(!value||value->t!=J_STR||strcmp(value->str,expected)){
        fprintf(stderr,"config.json: %s must be explicitly set to %s\n",
                key,expected);exit(1);
    }
}
static void q38_require_bool(jval *o,const char *key,int expected) {
    int value=q38_bool(o,key,expected);
    if(value!=expected){
        fprintf(stderr,"config.json: unsupported %s=%s (expected %s)\n",key,
                value?"true":"false",expected?"true":"false");exit(1);
    }
}

static void q38_load_cfg(Cfg *c,const char *snap) {
    memset(c,0,sizeof(*c));
    char path[2048]; snprintf(path,sizeof path,"%s/config.json",snap);
    FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);} fseek(f,0,SEEK_END);
    long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n<0||n>(256L<<20)){fprintf(stderr,"invalid config size\n");exit(1);}
    char *buf=(char*)malloc((size_t)n+1),*arena=NULL;
    if(!buf||fread(buf,1,(size_t)n,f)!=(size_t)n){fprintf(stderr,"cannot read %s\n",path);exit(1);}
    fclose(f); buf[n]=0; jval *root=json_parse(buf,&arena);
    if(!root||root->t!=J_OBJ){fprintf(stderr,"invalid %s\n",path);exit(1);}
    jval *tc=q38_obj(root,"text_config"); if(!tc) tc=root;
    const char *mt=jstr(tc,"model_type");
    if(!mt||strcmp(mt,"qwen4_exp_text")){fprintf(stderr,"unsupported text model_type: %s\n",mt?mt:"(missing)");exit(1);}
    q38_require_string(tc,"hidden_act","silu");
    /* Upstream's omitted/None default resolves to hidden_act (SiLU), whereas
     * this implementation deliberately uses the released sigmoid gate. */
    q38_require_present_string(tc,"output_gate_type","sigmoid");
    q38_require_bool(tc,"attention_bias",0);
    q38_require_bool(tc,"tie_word_embeddings",0);
    c->hidden=q38_num_int(tc,"hidden_size",0,1,0,INT_MAX);
    c->layers=q38_num_int(tc,"num_hidden_layers",0,1,0,Q38_MAX_LAYERS);
    c->vocab=q38_num_int(tc,"vocab_size",0,1,0,INT_MAX);
    c->max_positions=q38_num_int(tc,"max_position_embeddings",0,1,0,INT_MAX);
    c->eos_id=q38_num_int(tc,"eos_token_id",-1,1,INT_MIN,INT_MAX);
    double eps=q38_num(tc,"rms_norm_eps",1e-6,0);
    if(!isfinite(eps)||eps<=0.0||eps>FLT_MAX){fprintf(stderr,"invalid rms_norm_eps\n");exit(1);}
    c->eps=(float)eps;
    jval *rp=q38_obj(tc,"rope_parameters");
    if(rp)q38_require_string(rp,"rope_type","default");
    double theta=rp?q38_num(rp,"rope_theta",10000,0):q38_num(tc,"rope_theta",10000,0);
    if(!isfinite(theta)||theta<=0.0||theta>FLT_MAX){fprintf(stderr,"invalid rope_theta\n");exit(1);}
    c->theta=(float)theta;
    c->hc_count=q38_num_int(tc,"hc_count",4,0,0,INT_MAX);
    c->hc_rank=q38_num_int(tc,"hc_lowrank",320,0,0,INT_MAX);
    c->hc_width=q38_derived_product("hc_width",c->hc_count,c->hidden);
    c->q_heads=q38_num_int(tc,"num_attention_heads",0,1,0,INT_MAX);
    c->kv_heads=q38_num_int(tc,"num_key_value_heads",0,1,0,INT_MAX);
    c->head_dim=q38_num_int(tc,"head_dim",0,1,0,INT_MAX);
    double partial=rp?q38_num(rp,"partial_rotary_factor",q38_num(tc,"partial_rotary_factor",1,0),0):q38_num(tc,"partial_rotary_factor",1,0);
    double rotary_product=(double)c->head_dim*partial;
    if(!isfinite(partial)||!isfinite(rotary_product)||rotary_product<0||
       rotary_product>(double)INT_MAX){
        fprintf(stderr,"[qwen38 config] derived rotary_dim overflows int\n");exit(1);
    }
    /* Upstream Qwen4ExpTextConfig derives this with Python int(), i.e. truncation
     * toward zero rather than rounding to the nearest dimension. */
    c->rotary_dim=(int)rotary_product;
    c->idx_qheads=q38_num_int(tc,"indexer_n_heads",0,1,0,INT_MAX);
    c->idx_kheads=q38_num_int(tc,"indexer_kv_heads",0,1,0,INT_MAX);
    c->idx_dim=q38_num_int(tc,"indexer_head_dim",0,1,0,INT_MAX);
    c->idx_budget=q38_num_int(tc,"indexer_budget",0,1,0,INT_MAX);
    c->idx_ratio=q38_num_int(tc,"indexer_compress_ratio",0,1,0,INT_MAX);
    c->experts=q38_num_int(tc,"num_experts",0,1,0,INT_MAX);
    c->topk=q38_num_int(tc,"num_experts_per_tok",0,1,0,INT_MAX);
    c->inter=q38_num_int(tc,"moe_intermediate_size",0,1,0,INT_MAX);
    c->shared_inter=q38_num_int(tc,"shared_expert_intermediate_size",0,1,0,INT_MAX);
    c->norm_topk=q38_bool(tc,"norm_topk_prob",1);
    c->dn_kheads=q38_num_int(tc,"linear_num_key_heads",0,1,0,INT_MAX);
    c->dn_vheads=q38_num_int(tc,"linear_num_value_heads",0,1,0,INT_MAX);
    c->dn_kdim=q38_num_int(tc,"linear_key_head_dim",0,1,0,INT_MAX);
    c->dn_vdim=q38_num_int(tc,"linear_value_head_dim",0,1,0,INT_MAX);
    c->dn_convk=q38_num_int(tc,"linear_conv_kernel_dim",0,1,0,INT_MAX);
    int dn_qk=q38_derived_product("deltanet_qk",c->dn_kheads,c->dn_kdim);
    dn_qk=q38_derived_product("deltanet_qk_twice",2,dn_qk);
    c->dn_conv_dim=q38_derived_sum("dn_conv_dim",dn_qk,
                                    q38_derived_product("deltanet_v",c->dn_vheads,c->dn_vdim));
    c->ple_dim=q38_num_int(tc,"ple_embed_dim",c->hidden,0,0,INT_MAX);
    c->ple_convk=q38_num_int(tc,"ple_conv_kernel_size",4,0,0,INT_MAX);
    c->ngram_size=q38_num_int(tc,"ngram_size",3,0,0,INT_MAX);
    c->heads_per_ngram=q38_num_int(tc,"heads_per_ngram",8,0,0,INT_MAX);
    c->ngram_heads=(c->ngram_size>0)?q38_derived_product("ngram_heads",c->ngram_size-1,c->heads_per_ngram):0;
    if(c->ngram_heads<=0){fprintf(stderr,"[qwen38 config] derived ngram_heads invalid\n");exit(1);}
    c->ngram_head_dim=c->ple_dim/c->ngram_heads;
    c->ngram_parts=q38_num_int(tc,"split_ngram_parts",1,0,0,INT_MAX);
    c->ple_layer=-1;
    jval *pl=json_get(tc,"ple_layer_ids");
    if(pl&&pl->t==J_ARR&&pl->len){
        if(pl->len!=1||pl->kids[0]->t!=J_NUM){fprintf(stderr,"only one PLE layer is supported\n");exit(1);}
        double raw=pl->kids[0]->num;
        if(!isfinite(raw)||floor(raw)!=raw||raw<1||raw>(double)INT_MAX){
            fprintf(stderr,"config.json: ple_layer_ids[0] must be a positive integer\n");exit(1);
        }
        int one_based=(int)raw;
        c->ple_layer=one_based-1;
    }
    c->is_attn=(uint8_t*)calloc((size_t)c->layers,1);
    jval *lt=json_get(tc,"layer_types");
    if(!lt||lt->t!=J_ARR||lt->len!=c->layers){fprintf(stderr,"config layer_types must have %d entries\n",c->layers);exit(1);}
    for(int i=0;i<c->layers;i++){
        const char *s=lt->kids[i]->t==J_STR?lt->kids[i]->str:NULL;
        if(!s){fprintf(stderr,"invalid layer_types[%d]\n",i);exit(1);}
        if(!strcmp(s,"linear_attention")) c->is_attn[i]=0;
        else if(!strcmp(s,"full_attention")||!strcmp(s,"qwen_sparse_attention")) c->is_attn[i]=1;
        else {fprintf(stderr,"unsupported layer type %s\n",s);exit(1);}
    }
    json_free(root); free(buf); free(arena);
}

#define Q38_NEED(x,...) do{if(!(x)){fprintf(stderr,"[qwen38 config] ");fprintf(stderr,__VA_ARGS__);fprintf(stderr," -- refusing\n");exit(1);}}while(0)
static void q38_validate_cfg(const Cfg *c) {
    Q38_NEED(c->hidden>0&&c->hidden<=65536,"hidden_size=%d",c->hidden);
    Q38_NEED(c->layers>0&&c->layers<=Q38_MAX_LAYERS,"layers=%d",c->layers);
    Q38_NEED(c->vocab>0&&c->max_positions>0&&
             c->max_positions<=QWEN38_ATTN_MAX_CTX,"vocab/context invalid");
    Q38_NEED(isfinite(c->eps)&&c->eps>0.f&&isfinite(c->theta)&&c->theta>0.f,"RoPE/norm constants invalid");
    Q38_NEED(c->eos_id>=0&&c->eos_id<c->vocab,"eos=%d",c->eos_id);
    Q38_NEED(c->hc_count>1&&c->hc_count<=16&&c->hc_rank>0&&
             c->hc_rank<=65536&&c->hc_width>0,"gated residual dimensions invalid");
    Q38_NEED(c->q_heads>0&&c->kv_heads>0&&c->q_heads%c->kv_heads==0,"attention heads invalid");
    Q38_NEED(c->head_dim>0&&c->rotary_dim>0&&!(c->rotary_dim&1)&&c->rotary_dim<=c->head_dim,"RoPE dimensions invalid");
    Q38_NEED(c->head_dim<=INT_MAX/2&&c->q_heads<=INT_MAX/(2*c->head_dim)&&
             c->kv_heads<=INT_MAX/c->head_dim,"attention projection dimensions overflow");
    Q38_NEED(c->idx_qheads>0&&c->idx_kheads==1&&c->idx_dim>=c->rotary_dim,"indexer dimensions invalid");
    Q38_NEED(c->idx_ratio>0&&c->idx_budget>0&&c->idx_budget%c->idx_ratio==0&&
             c->idx_budget<=INT_MAX-c->idx_ratio+1,"indexer budget invalid");
    Q38_NEED(c->idx_qheads<INT_MAX&&c->idx_qheads<=INT_MAX/c->idx_dim-c->idx_kheads,
             "indexer projection dimensions overflow");
    Q38_NEED(c->experts>0&&c->experts<=Q38_MAX_EXPERTS,"experts=%d",c->experts);
    Q38_NEED(c->topk>0&&c->topk<=Q38_MAX_TOPK&&c->topk<=c->experts,"topk=%d",c->topk);
    Q38_NEED(c->inter<=INT_MAX/2,"expert projection dimensions overflow");
    Q38_NEED(c->inter>0&&c->shared_inter>0,"MoE widths invalid");
    Q38_NEED(c->dn_vheads>0&&c->dn_kheads>0&&c->dn_vheads%c->dn_kheads==0,"DeltaNet heads invalid");
    Q38_NEED(c->dn_kdim>0&&c->dn_vdim>0&&c->dn_vdim<=512&&c->dn_convk>=2&&c->dn_conv_dim>0&&
             c->dn_vdim<=INT_MAX/c->dn_vheads&&c->dn_kdim<=INT_MAX/c->dn_vdim,
             "DeltaNet dimensions invalid");
    Q38_NEED((uint64_t)c->dn_vheads<=SIZE_MAX/sizeof(float)/(uint64_t)c->dn_kdim/(uint64_t)c->dn_vdim&&
             (uint64_t)c->dn_conv_dim<=SIZE_MAX/sizeof(float)/(uint64_t)(c->dn_convk-1),
             "DeltaNet state dimensions overflow");
    Q38_NEED(c->ngram_size==3&&c->ngram_heads>0&&c->ngram_heads<=64&&
             c->ple_dim>0&&c->ple_dim%c->ngram_heads==0&&
             c->ngram_head_dim>0&&c->ngram_head_dim<=512&&
             c->ple_convk>=2&&c->ple_convk<=INT_MAX/c->ngram_size,
             "PLE dimensions invalid");
    Q38_NEED((uint64_t)c->hc_width<=SIZE_MAX/sizeof(float)/(uint64_t)(c->ple_convk-1)/(uint64_t)c->ngram_size,
             "PLE state dimensions overflow");
    Q38_NEED(c->ngram_parts>0&&c->ngram_parts<=Q38_MAX_PLE_PARTS,"PLE parts=%d",c->ngram_parts);
    Q38_NEED(c->ple_layer>=0&&c->ple_layer<c->layers,"PLE layer=%d",c->ple_layer);
}

static float *q38_load_tensor(Model *m,const char *name,int64_t want) {
    st_tensor *t=st_find(&m->S,name);
    if(!t){fprintf(stderr,"missing %s\n",name);exit(1);}
    if(t->numel!=want){fprintf(stderr,"%s: %lld elements, expected %lld\n",name,(long long)t->numel,(long long)want);exit(1);}
    float *p=falloc(want); st_read_f32(&m->S,name,p,1); return p;
}

static void q38_name(Model *m,char *out,size_t cap,int layer,const char *suffix) {
    snprintf(out,cap,"%s.layers.%d.%s",m->prefix,layer,suffix);
}

static void q38_load_gr(Model *m,GatedResidual *g,int layer,const char *kind,int inject) {
    Cfg *c=&m->c; char nm[320],base[180];
    if(layer>=0) snprintf(base,sizeof base,"layers.%d.%s",layer,kind); else snprintf(base,sizeof base,"hyper_connection_mixer");
    snprintf(nm,sizeof nm,"%s.%s.hc_norm.weight",m->prefix,base); g->norm=q38_load_tensor(m,nm,c->hc_width);
    snprintf(nm,sizeof nm,"%s.%s.input_mix_weight_down.weight",m->prefix,base); g->down=q38_load_tensor(m,nm,(int64_t)c->hc_rank*c->hc_width);
    snprintf(nm,sizeof nm,"%s.%s.input_mix_weight_up.weight",m->prefix,base); g->up=q38_load_tensor(m,nm,(int64_t)c->hc_width*c->hc_rank);
    if(inject){snprintf(nm,sizeof nm,"%s.%s.block_inject_weight.weight",m->prefix,base);g->inject=q38_load_tensor(m,nm,(int64_t)c->hc_count*c->hc_width);}
}

static void q38_load_ple(Model *m,Layer *l) {
    Cfg *c=&m->c; int i=c->ple_layer; char nm[320];
    #define PL(field,suf,n) q38_name(m,nm,sizeof nm,i,"ple." suf); l->field=q38_load_tensor(m,nm,(n))
    PL(ple_key,"key_proj.weight",(int64_t)c->hc_width*c->ple_dim);
    PL(ple_value,"value_proj.weight",(int64_t)c->hidden*c->ple_dim);
    PL(ple_norm_key,"norm_key.weight",c->hc_width);
    PL(ple_norm_query,"norm_query.weight",c->hc_width);
    PL(ple_norm_conv,"norm_conv.weight",c->hc_width);
    PL(ple_conv,"conv1d.weight",(int64_t)c->hc_width*c->ple_convk);
    #undef PL
    const char *bufs[]={"layer_multipliers","ngram_heads_vocab_sizes","ngram_heads_offsets"};
    int64_t *dsts[]={m->ple_multipliers,m->ple_head_vocab,m->ple_head_offset};
    int counts[]={c->ngram_size,c->ngram_heads,c->ngram_heads};
    for(int b=0;b<3;b++){
        q38_name(m,nm,sizeof nm,i,"ple.ple_embedding."); strncat(nm,bufs[b],sizeof(nm)-strlen(nm)-1);
        st_tensor *t=st_find(&m->S,nm); if(!t||t->dtype!=6||t->numel!=counts[b]){fprintf(stderr,"invalid %s\n",nm);exit(1);}
        st_read_raw_cap(&m->S,nm,dsts[b],(int64_t)counts[b]*8,1);
    }
    for(int h=0;h<c->ngram_heads;h++)
        Q38_NEED(m->ple_head_vocab[h]>0&&m->ple_head_offset[h]>=0&&
                 m->ple_head_vocab[h]<=INT64_MAX-m->ple_head_offset[h],
                 "invalid PLE head range %d",h);
    m->ple_part_count=0; m->ple_part_start[0]=0;
    q38_name(m,nm,sizeof nm,i,"ple.ple_embedding.ngram_embedding.weight");
    if(st_has(&m->S,nm)){
        st_tensor *t=st_find(&m->S,nm);
        if(!t||t->rank!=2||t->shape[0]<=0||t->shape[1]!=c->ngram_head_dim){fprintf(stderr,"invalid PLE table %s\n",nm);exit(1);}
        snprintf(m->ple_part_names[0],sizeof m->ple_part_names[0],"%s",nm);
        m->ple_parts[0]=t; m->ple_part_count=1;
        m->ple_part_start[1]=m->ple_parts[0]->shape[0];
    } else {
        for(int p=0;p<c->ngram_parts;p++){
            q38_name(m,nm,sizeof nm,i,"ple.ple_embedding.ngram_embedding.");
            size_t z=strlen(nm); snprintf(nm+z,sizeof(nm)-z,"shard_%d.weight",p);
            st_tensor *t=st_find(&m->S,nm); if(!t){fprintf(stderr,"missing PLE shard %s\n",nm);exit(1);}
            if(t->rank!=2||t->shape[0]<=0||t->shape[1]!=c->ngram_head_dim||
               m->ple_part_start[p]>INT64_MAX-t->shape[0]){fprintf(stderr,"invalid PLE shard %s\n",nm);exit(1);}
            snprintf(m->ple_part_names[p],sizeof m->ple_part_names[p],"%s",nm);
            m->ple_parts[p]=t; m->ple_part_start[p+1]=m->ple_part_start[p]+t->shape[0];
            m->ple_part_count++;
        }
    }
    q38_name(m,nm,sizeof nm,i,"ple.ple_embedding.ngram_embedding.weight_scale");
    m->ple_weight_scale=1.f;
    if(st_has(&m->S,nm)){
        st_tensor *t=st_find(&m->S,nm); if(t->numel!=1){fprintf(stderr,"invalid %s\n",nm);exit(1);}
        st_read_f32(&m->S,nm,&m->ple_weight_scale,1);
    }
    int64_t need=0; for(int h=0;h<c->ngram_heads;h++) if(m->ple_head_offset[h]+m->ple_head_vocab[h]>need) need=m->ple_head_offset[h]+m->ple_head_vocab[h];
    Q38_NEED(m->ple_part_start[m->ple_part_count]>=need,"PLE table rows %lld < required %lld",(long long)m->ple_part_start[m->ple_part_count],(long long)need);
}

static void q38_alloc_state(Model *m) {
    Cfg *c=&m->c;
    m->DN_rec=(float**)calloc((size_t)c->layers,sizeof(float*));
    m->DN_conv=(float**)calloc((size_t)c->layers,sizeof(float*));
    m->K=(float**)calloc((size_t)c->layers,sizeof(float*));
    m->V=(float**)calloc((size_t)c->layers,sizeof(float*));
    m->IK=(float**)calloc((size_t)c->layers,sizeof(float*));
    if(!m->DN_rec||!m->DN_conv||!m->K||!m->V||!m->IK){fprintf(stderr,"OOM model state metadata\n");exit(1);}
    for(int i=m->range_begin;i<m->range_end;i++) if(!c->is_attn[i]) {
        m->DN_rec[i]=(float*)calloc((size_t)c->dn_vheads*c->dn_kdim*c->dn_vdim,sizeof(float));
        m->DN_conv[i]=(float*)calloc((size_t)c->dn_conv_dim*(c->dn_convk-1),sizeof(float));
        if(!m->DN_rec[i]||!m->DN_conv[i]){fprintf(stderr,"OOM DeltaNet state\n");exit(1);}
    }
    if(c->ple_layer>=m->range_begin&&c->ple_layer<m->range_end){
        m->ple_history=(int64_t*)calloc(2,sizeof(int64_t));
        m->PLE_conv_state=(float*)calloc((size_t)c->hc_width*(c->ple_convk-1)*c->ngram_size,sizeof(float));
        if(!m->ple_history||!m->PLE_conv_state){fprintf(stderr,"OOM PLE state\n");exit(1);}
    }
}

static void model_init_range(Model *m,const char *snap,int cap,int bits,
                             int layer_begin,int layer_end,int load_boundaries,
                             int allocate_state) {
    (void)bits; memset(m,0,sizeof(*m)); double t0=now_s();
    q38_load_cfg(&m->c,snap); q38_validate_cfg(&m->c); st_init(&m->S,snap);
    Cfg *c=&m->c; char nm[320];
    if(st_has(&m->S,"model.language_model.embed_tokens.weight")) snprintf(m->prefix,sizeof m->prefix,"model.language_model");
    else if(st_has(&m->S,"model.embed_tokens.weight")) snprintf(m->prefix,sizeof m->prefix,"model");
    else {fprintf(stderr,"checkpoint has no Qwen4-Exp text embedding\n");exit(1);}
    if(layer_end==0) layer_end=c->layers;
    if(layer_begin<0||layer_begin>=layer_end||layer_end>c->layers){fprintf(stderr,"invalid Qwen3.8 layer range [%d,%d)\n",layer_begin,layer_end);exit(1);}
    m->range_begin=layer_begin; m->range_end=layer_end;
    if(load_boundaries){
        snprintf(nm,sizeof nm,"%s.embed_tokens.weight",m->prefix); m->embed=q38_load_tensor(m,nm,(int64_t)c->vocab*c->hidden);
        m->lm_head=q38_load_tensor(m,"lm_head.weight",(int64_t)c->vocab*c->hidden);
        q38_load_gr(m,&m->final_gr,-1,NULL,0);
    }
    m->L=(Layer*)calloc((size_t)c->layers,sizeof(Layer));
    m->cache=(LCache*)calloc((size_t)c->layers,sizeof(LCache));
    if(!m->L||!m->cache){fprintf(stderr,"OOM model metadata\n");exit(1);}
    for(int i=layer_begin;i<layer_end;i++){
        Layer *l=&m->L[i]; q38_load_gr(m,&l->attn_gr,i,"attn_hyper_connection",1); q38_load_gr(m,&l->mlp_gr,i,"mlp_hyper_connection",1);
        #define LD(field,suf,n) q38_name(m,nm,sizeof nm,i,suf); l->field=q38_load_tensor(m,nm,(n))
        LD(router,"mlp.gate.weight",(int64_t)c->experts*c->hidden);
        LD(sh_g,"mlp.shared_expert.gate_proj.weight",(int64_t)c->shared_inter*c->hidden);
        LD(sh_u,"mlp.shared_expert.up_proj.weight",(int64_t)c->shared_inter*c->hidden);
        LD(sh_d,"mlp.shared_expert.down_proj.weight",(int64_t)c->hidden*c->shared_inter);
        LD(sh_gate,"mlp.shared_expert_gate.weight",c->hidden);
        if(c->is_attn[i]){
            LD(q,"self_attn.q_proj.weight",(int64_t)c->q_heads*c->head_dim*2*c->hidden);
            LD(k,"self_attn.k_proj.weight",(int64_t)c->kv_heads*c->head_dim*c->hidden);
            LD(v,"self_attn.v_proj.weight",(int64_t)c->kv_heads*c->head_dim*c->hidden);
            LD(o,"self_attn.o_proj.weight",(int64_t)c->hidden*c->q_heads*c->head_dim);
            LD(qn,"self_attn.q_norm.weight",c->head_dim);
            LD(kn,"self_attn.k_norm.weight",c->head_dim);
            LD(idx_qk,"self_attn.indexer.index_qk_proj.weight",(int64_t)(c->idx_qheads+c->idx_kheads)*c->idx_dim*c->hidden);
            LD(idx_qn,"self_attn.indexer.q_layernorm.weight",c->idx_dim);
            LD(idx_kn,"self_attn.indexer.k_layernorm.weight",c->idx_dim);
        } else {
            int vd=c->dn_vheads*c->dn_vdim;
            LD(dn_qkv,"linear_attn.in_proj_qkv.weight",(int64_t)c->dn_conv_dim*c->hidden);
            LD(dn_z,"linear_attn.in_proj_z.weight",(int64_t)vd*c->hidden);
            LD(dn_b,"linear_attn.in_proj_b.weight",(int64_t)c->dn_vheads*c->hidden);
            LD(dn_a,"linear_attn.in_proj_a.weight",(int64_t)c->dn_vheads*c->hidden);
            LD(dn_conv,"linear_attn.conv1d.weight",(int64_t)c->dn_conv_dim*c->dn_convk);
            LD(dn_dtbias,"linear_attn.dt_bias",c->dn_vheads);
            LD(dn_alog,"linear_attn.A_log",c->dn_vheads);
            LD(dn_norm,"linear_attn.norm.weight",c->dn_vdim);
            LD(dn_out,"linear_attn.out_proj.weight",(int64_t)c->hidden*vd);
        }
        #undef LD
        LCache *lc=&m->cache[i]; lc->cap=cap; lc->slots=(Slot*)calloc((size_t)cap,sizeof(Slot)); lc->by_expert=(int*)malloc((size_t)c->experts*sizeof(int));
        if(!lc->slots||!lc->by_expert){fprintf(stderr,"OOM expert cache\n");exit(1);} for(int e=0;e<c->experts;e++)lc->by_expert[e]=-1;
    }
    if(c->ple_layer>=layer_begin&&c->ple_layer<layer_end) q38_load_ple(m,&m->L[c->ple_layer]);
    if(allocate_state) q38_alloc_state(m);
    m->dense_load_s=now_s()-t0;
    fprintf(stderr,"[qwen38] native text weights: prefix=%s, %d layers, PLE=%d, cache=%d/layer\n",m->prefix,c->layers,c->ple_layer,cap);
}

static void model_init(Model *m,const char *snap,int cap,int bits) {
    model_init_range(m,snap,cap,bits,0,0,1,1);
}

static void q38_gr_read(const Cfg *c,const GatedResidual *g,const float *hyper,
                        int S,float *mixed,float *inject) {
    int H=c->hidden,W=c->hc_width,C=c->hc_count,R=c->hc_rank;
    float *norm=falloc((int64_t)S*W),*low=falloc((int64_t)S*R),*mix=falloc((int64_t)S*W);
    for(int s=0;s<S;s++) for(int b=0;b<C;b++)
        q38_rms0(norm+(int64_t)s*W+(int64_t)b*H,hyper+(int64_t)s*W+(int64_t)b*H,g->norm+(int64_t)b*H,H,c->eps);
    q38_matmul(low,norm,g->down,S,W,R);
    for(int64_t z=0;z<(int64_t)S*R;z++) low[z]=q38_silu(low[z]/C);
    q38_matmul(mix,low,g->up,S,R,W);
    for(int s=0;s<S;s++) for(int d=0;d<H;d++){
        float v=0.f;
        for(int b=0;b<C;b++) v+=q38_sigmoid(mix[(int64_t)s*W+(int64_t)b*H+d])*norm[(int64_t)s*W+(int64_t)b*H+d];
        mixed[(int64_t)s*H+d]=v/C;
    }
    if(inject){
        q38_matmul(inject,norm,g->inject,S,W,C);
        for(int64_t z=0;z<(int64_t)S*C;z++) inject[z]=2.f*q38_sigmoid(inject[z]/C);
    }
    free(norm);free(low);free(mix);
}

static void q38_gr_apply(const Cfg *c,float *hyper,const float *block,const float *inject,int S) {
    int H=c->hidden,C=c->hc_count,W=c->hc_width;
    for(int s=0;s<S;s++)for(int b=0;b<C;b++){
        float a=inject[(int64_t)s*C+b];
        for(int d=0;d<H;d++)hyper[(int64_t)s*W+(int64_t)b*H+d]+=a*block[(int64_t)s*H+d];
    }
}

static void q38_decode_fp8(Model *m,const char *wn,const char *sn,float *out,int O,int I) {
    st_tensor *w=st_find(&m->S,wn),*sc=st_find(&m->S,sn);
    int nb_o=(O+127)/128,nb_i=(I+127)/128;
    if(!w||w->dtype!=4||w->rank!=2||w->shape[0]!=O||w->shape[1]!=I||
       !sc||sc->rank!=2||sc->shape[0]!=nb_o||sc->shape[1]!=nb_i){
        fprintf(stderr,"invalid block-FP8 expert matrix %s / %s\n",wn,sn);exit(1);
    }
    uint8_t *raw=(uint8_t*)malloc((size_t)O*I); float *scale=falloc((int64_t)nb_o*nb_i);
    if(!raw){fprintf(stderr,"OOM FP8 expert read\n");exit(1);}
    st_read_raw_cap(&m->S,wn,raw,(int64_t)O*I,1); st_read_f32(&m->S,sn,scale,1);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++) for(int i=0;i<I;i++) out[(int64_t)o*I+i]=e4m3_decode(raw[(int64_t)o*I+i])*scale[(o/128)*nb_i+i/128];
    free(raw);free(scale);
}

static void q38_load_expert(Model *m,int layer,int eid,Slot *s) {
    Cfg *c=&m->c; int H=c->hidden,I=c->inter; char nm[320],sn[340];
    if(!s->gate){
        float *all=falloc((int64_t)3*I*H); s->gate=all;s->up=all+(int64_t)I*H;s->down=all+(int64_t)2*I*H;
    }
    q38_name(m,nm,sizeof nm,layer,"mlp.experts.gate_up_proj");
    if(st_has(&m->S,nm)){
        st_tensor *t=st_find(&m->S,nm);
        if(t->rank!=3||t->shape[0]!=c->experts||t->shape[1]!=2*I||t->shape[2]!=H){fprintf(stderr,"invalid %s\n",nm);exit(1);}
        st_read_slice_f32(&m->S,nm,(int64_t)eid*2*I*H,(int64_t)2*I*H,s->gate,1);
        q38_name(m,nm,sizeof nm,layer,"mlp.experts.down_proj"); t=st_find(&m->S,nm);
        if(!t||t->rank!=3||t->shape[0]!=c->experts||t->shape[1]!=H||t->shape[2]!=I){fprintf(stderr,"invalid %s\n",nm);exit(1);}
        st_read_slice_f32(&m->S,nm,(int64_t)eid*H*I,(int64_t)H*I,s->down,1);
        return;
    }
    const char *kind[3]={"gate_proj","up_proj","down_proj"}; float *dst[3]={s->gate,s->up,s->down};
    int os[3]={I,I,H},is[3]={H,H,I};
    for(int k=0;k<3;k++){
        char suf[192]; snprintf(suf,sizeof suf,"mlp.experts.%d.%s.weight",eid,kind[k]);q38_name(m,nm,sizeof nm,layer,suf);
        st_tensor *t=st_find(&m->S,nm); if(!t){fprintf(stderr,"missing %s\n",nm);exit(1);}
        if(t->dtype==4){snprintf(sn,sizeof sn,"%s_scale_inv",nm);q38_decode_fp8(m,nm,sn,dst[k],os[k],is[k]);}
        else {
            if(t->rank!=2||t->shape[0]!=os[k]||t->shape[1]!=is[k]){fprintf(stderr,"invalid %s\n",nm);exit(1);}
            st_read_f32(&m->S,nm,dst[k],1);
        }
    }
}

static Slot *q38_expert_get(Model *m,int layer,int eid) {
    LCache *lc=&m->cache[layer]; int si=lc->by_expert[eid];
    if(si>=0){m->hits++;lc->slots[si].used=++m->clock;return &lc->slots[si];}
    m->miss++; Slot *s;
    if(lc->n<lc->cap){s=&lc->slots[lc->n++];s->eid=-1;}
    else {
        int victim=0;for(int i=1;i<lc->n;i++)if(lc->slots[i].used<lc->slots[victim].used)victim=i;
        s=&lc->slots[victim];if(s->eid>=0)lc->by_expert[s->eid]=-1;
    }
    s->eid=-1;q38_load_expert(m,layer,eid,s);s->eid=eid;s->used=++m->clock;lc->by_expert[eid]=(int)(s-lc->slots);return s;
}

static void q38_ple_row(Model *m,int64_t row,float *out) {
    Cfg *c=&m->c; int p=0;
    while(p+1<m->ple_part_count&&row>=m->ple_part_start[p+1])p++;
    if(p>=m->ple_part_count||row<m->ple_part_start[p]){fprintf(stderr,"PLE row out of range: %lld\n",(long long)row);exit(1);}
    int64_t local=row-m->ple_part_start[p]; st_tensor *t=m->ple_parts[p]; const char *nm=m->ple_part_names[p];
    Q38_NEED(c->ngram_head_dim>0&&local>=0&&local<=INT64_MAX/c->ngram_head_dim,
             "PLE row byte offset overflows");
    if(t->dtype==4){
        uint8_t raw[512]; Q38_NEED(c->ngram_head_dim<=(int)sizeof raw,"PLE row too wide");
        st_read_slice_raw_cap(&m->S,nm,local*c->ngram_head_dim,c->ngram_head_dim,raw,sizeof raw,1);
        for(int d=0;d<c->ngram_head_dim;d++)out[d]=e4m3_decode(raw[d])*m->ple_weight_scale;
    } else st_read_slice_f32(&m->S,nm,local*c->ngram_head_dim,c->ngram_head_dim,out,1);
}

static int64_t q38_hash_row(Model *m,int head,int ngram,int64_t cur,int64_t p1,int64_t p2) {
    uint64_t x=(uint64_t)cur*(uint64_t)m->ple_multipliers[0];
    x^=(uint64_t)p1*(uint64_t)m->ple_multipliers[1];
    if(ngram==3)x^=(uint64_t)p2*(uint64_t)m->ple_multipliers[2];
    int64_t sx=(int64_t)x,mod=m->ple_head_vocab[head],r=sx%mod;if(r<0)r+=mod;
    return m->ple_head_offset[head]+r;
}

static void q38_ple(Model *m,const int *ids,int S,const float *hyper,float *out) {
    Cfg *c=&m->c; Layer *l=&m->L[c->ple_layer]; int H=c->hidden,C=c->hc_count,W=c->hc_width,E=c->ple_dim;
    float *emb=falloc(E),*keys=falloc(W),*value=falloc(H),*kn=falloc(W),*qn=falloc(W),*gated=falloc(W),*norm=falloc(W);
    int state_len=(c->ple_convk-1)*c->ngram_size; float *ring=m->PLE_conv_state;
    for(int s=0;s<S;s++){
        int64_t p1=m->ple_history_len>=1?m->ple_history[m->ple_history_len-1]:c->eos_id;
        int64_t p2=m->ple_history_len>=2?m->ple_history[m->ple_history_len-2]:c->eos_id;
        for(int h=0;h<c->ngram_heads;h++){
            int ng=h<c->heads_per_ngram?2:3; int64_t row=q38_hash_row(m,h,ng,ids[s],p1,p2);
            q38_ple_row(m,row,emb+(int64_t)h*c->ngram_head_dim);
        }
        q38_matmul(keys,emb,l->ple_key,1,E,W);q38_matmul(value,emb,l->ple_value,1,E,H);
        for(int b=0;b<C;b++){
            q38_rms0(kn+(int64_t)b*H,keys+(int64_t)b*H,l->ple_norm_key+(int64_t)b*H,H,c->eps);
            q38_rms0(qn+(int64_t)b*H,hyper+(int64_t)s*W+(int64_t)b*H,l->ple_norm_query+(int64_t)b*H,H,c->eps);
            float dot=0.f;for(int d=0;d<H;d++)dot+=kn[(int64_t)b*H+d]*qn[(int64_t)b*H+d];dot/=sqrtf((float)H);
            float shaped=copysignf(sqrtf(fmaxf(fabsf(dot),1e-6f)),dot),g=q38_sigmoid(shaped);
            for(int d=0;d<H;d++)gated[(int64_t)b*H+d]=g*value[d];
            q38_rms0(norm+(int64_t)b*H,gated+(int64_t)b*H,l->ple_norm_conv+(int64_t)b*H,H,c->eps);
        }
        for(int d=0;d<W;d++){
            float a=l->ple_conv[(int64_t)d*c->ple_convk+c->ple_convk-1]*norm[d];
            for(int k=0;k<c->ple_convk-1;k++)a+=l->ple_conv[(int64_t)d*c->ple_convk+k]*ring[(int64_t)d*state_len+k*c->ngram_size];
            out[(int64_t)s*W+d]=gated[d]+q38_silu(a);
            float *r=ring+(int64_t)d*state_len;for(int k=0;k<state_len-1;k++)r[k]=r[k+1];r[state_len-1]=norm[d];
        }
        if(ids[s]==c->eos_id)m->ple_history_len=0;
        else if(m->ple_history_len==0){m->ple_history[0]=ids[s];m->ple_history_len=1;}
        else if(m->ple_history_len==1){m->ple_history[1]=ids[s];m->ple_history_len=2;}
        else {m->ple_history[0]=m->ple_history[1];m->ple_history[1]=ids[s];}
    }
    free(emb);free(keys);free(value);free(kn);free(qn);free(gated);free(norm);
}

static void q38_deltanet(Model *m,Layer *l,int layer,const float *x,int S,float *out) {
    Cfg *c=&m->c; int H=c->hidden,VH=c->dn_vheads,KH=c->dn_kheads,KD=c->dn_kdim,VD=c->dn_vdim;
    int CD=c->dn_conv_dim,CK=c->dn_convk,V=VH*VD,K=KH*KD,rep=VH/KH;
    float *qkv=falloc(CD),*z=falloc(V),*bb=falloc(VH),*aa=falloc(VH),*conv=falloc(CD);
    float *q=falloc((int64_t)VH*KD),*k=falloc((int64_t)VH*KD),*core=falloc(V),*norm=falloc(V);
    float *rec=m->DN_rec[layer],*ring=m->DN_conv[layer];
    for(int s=0;s<S;s++){
        const float *xs=x+(int64_t)s*H;
        q38_matmul(qkv,xs,l->dn_qkv,1,H,CD);q38_matmul(z,xs,l->dn_z,1,H,V);
        q38_matmul(bb,xs,l->dn_b,1,H,VH);q38_matmul(aa,xs,l->dn_a,1,H,VH);
        for(int d=0;d<CD;d++){
            float a=l->dn_conv[(int64_t)d*CK+CK-1]*qkv[d];float *r=ring+(int64_t)d*(CK-1);
            for(int j=0;j<CK-1;j++)a+=l->dn_conv[(int64_t)d*CK+j]*r[j];conv[d]=q38_silu(a);
            for(int j=0;j<CK-2;j++)r[j]=r[j+1];r[CK-2]=qkv[d];
        }
        const float *qi=conv,*ki=conv+K,*vi=conv+2*K;
        for(int h=0;h<VH;h++){
            memcpy(q+(int64_t)h*KD,qi+(int64_t)(h/rep)*KD,(size_t)KD*sizeof(float));
            memcpy(k+(int64_t)h*KD,ki+(int64_t)(h/rep)*KD,(size_t)KD*sizeof(float));
            double qs=1e-6,ks=1e-6;for(int d=0;d<KD;d++){qs+=(double)q[(int64_t)h*KD+d]*q[(int64_t)h*KD+d];ks+=(double)k[(int64_t)h*KD+d]*k[(int64_t)h*KD+d];}
            float qr=1.f/sqrtf((float)qs)/sqrtf((float)KD),kr=1.f/sqrtf((float)ks);
            for(int d=0;d<KD;d++){q[(int64_t)h*KD+d]*=qr;k[(int64_t)h*KD+d]*=kr;}
        }
        #pragma omp parallel for schedule(static)
        for(int h=0;h<VH;h++){
            float *state=rec+(int64_t)h*KD*VD;const float *qh=q+(int64_t)h*KD,*kh=k+(int64_t)h*KD,*vh=vi+(int64_t)h*VD;
            float alpha=expf(-expf(l->dn_alog[h])*q38_softplus(aa[h]+l->dn_dtbias[h]));
            float beta=q38_sigmoid(bb[h]);float delta[512];
            int64_t state_cells=(int64_t)KD*VD;
            for(int64_t z0=0;z0<state_cells;z0++)state[z0]*=alpha;
            for(int v0=0;v0<VD;v0++){float a=0.f;for(int d=0;d<KD;d++)a+=kh[d]*state[(int64_t)d*VD+v0];delta[v0]=(vh[v0]-a)*beta;}
            for(int d=0;d<KD;d++)for(int v0=0;v0<VD;v0++)state[(int64_t)d*VD+v0]+=kh[d]*delta[v0];
            for(int v0=0;v0<VD;v0++){float a=0.f;for(int d=0;d<KD;d++)a+=qh[d]*state[(int64_t)d*VD+v0];core[(int64_t)h*VD+v0]=a;}
        }
        for(int h=0;h<VH;h++)q38_rmsg(norm+(int64_t)h*VD,core+(int64_t)h*VD,z+(int64_t)h*VD,l->dn_norm,VD,c->eps,1);
        q38_matmul(out+(int64_t)s*H,norm,l->dn_out,1,V,H);
    }
    free(qkv);free(z);free(bb);free(aa);free(conv);free(q);free(k);free(core);free(norm);
}

typedef struct { float score; int block; } Q38Block;
static int q38_block_desc(const void *aa,const void *bb){
    const Q38Block *a=(const Q38Block*)aa,*b=(const Q38Block*)bb;
    if(a->score>b->score)return -1;if(a->score<b->score)return 1;return a->block-b->block;
}

static void q38_attention(Model *m,Layer *l,int layer,const float *x,int S,int pos_base,float *out) {
    Cfg *c=&m->c;int H=c->hidden,QH=c->q_heads,KVH=c->kv_heads,D=c->head_dim;
    int IQ=c->idx_qheads,ID=c->idx_dim,R=c->idx_ratio,maxsel=c->idx_budget+R-1;
    float *qp=falloc((int64_t)S*QH*2*D),*kp=falloc((int64_t)S*KVH*D),*vp=falloc((int64_t)S*KVH*D);
    float *ip=falloc((int64_t)S*(IQ+c->idx_kheads)*ID);
    q38_matmul(qp,x,l->q,S,H,QH*2*D);q38_matmul(kp,x,l->k,S,H,KVH*D);q38_matmul(vp,x,l->v,S,H,KVH*D);
    q38_matmul(ip,x,l->idx_qk,S,H,(IQ+c->idx_kheads)*ID);
    for(int s=0;s<S;s++){
        int pos=pos_base+s;
        for(int h=0;h<KVH;h++){
            float *kh=kp+(int64_t)s*KVH*D+(int64_t)h*D;q38_rms0(kh,kh,l->kn,D,c->eps);q38_rope(kh,D,c->rotary_dim,pos,c->theta);
            memcpy(m->K[layer]+((int64_t)h*m->kv_cap+pos)*D,kh,(size_t)D*sizeof(float));
            memcpy(m->V[layer]+((int64_t)h*m->kv_cap+pos)*D,vp+(int64_t)s*KVH*D+(int64_t)h*D,(size_t)D*sizeof(float));
        }
        memcpy(m->IK[layer]+(int64_t)pos*ID,ip+(int64_t)s*(IQ+1)*ID+(int64_t)IQ*ID,(size_t)ID*sizeof(float));
    }
    float *heads=falloc((int64_t)S*QH*D),*qidx=falloc((int64_t)IQ*ID),*pool=falloc(ID);
    int *selected=(int*)malloc((size_t)maxsel*sizeof(int));
    if(!selected){fprintf(stderr,"OOM QSA selection\n");exit(1);}
    for(int s=0;s<S;s++){
        int pos=pos_base+s,visible=pos+1,blocks=visible/R,tail=blocks*R;
        for(int h=0;h<IQ;h++){float *qh=qidx+(int64_t)h*ID;memcpy(qh,ip+(int64_t)s*(IQ+1)*ID+(int64_t)h*ID,(size_t)ID*sizeof(float));q38_rms0(qh,qh,l->idx_qn,ID,c->eps);q38_rope(qh,ID,c->rotary_dim,pos,c->theta);}
        int take=blocks<c->idx_budget/R?blocks:c->idx_budget/R,nsel=0;
        Q38Block *rank=blocks?(Q38Block*)malloc((size_t)blocks*sizeof(Q38Block)):NULL;
        if(blocks&&!rank){fprintf(stderr,"OOM QSA block ranking\n");exit(1);}
        for(int b=0;b<blocks;b++){
            memset(pool,0,(size_t)ID*sizeof(float));for(int r=0;r<R;r++){const float *raw=m->IK[layer]+(int64_t)(b*R+r)*ID;for(int d=0;d<ID;d++)pool[d]+=raw[d]/R;}
            q38_rms0(pool,pool,l->idx_kn,ID,c->eps);q38_rope(pool,ID,c->rotary_dim,b*R,c->theta);
            float score=0.f;for(int h=0;h<IQ;h++){float a=0.f;for(int d=0;d<ID;d++)a+=qidx[(int64_t)h*ID+d]*pool[d];if(a>0.f)score+=a;}rank[b]=(Q38Block){score/sqrtf((float)ID),b};
        }
        if(blocks)qsort(rank,(size_t)blocks,sizeof(Q38Block),q38_block_desc);
        for(int z=0;z<take;z++)for(int r=0;r<R;r++)selected[nsel++]=rank[z].block*R+r;
        for(int t=tail;t<visible;t++)selected[nsel++]=t;free(rank);
        for(int h=0;h<QH;h++){
            float *qraw=qp+(int64_t)s*QH*2*D+(int64_t)h*2*D;
            float *qh=falloc(D);memcpy(qh,qraw,(size_t)D*sizeof(float));q38_rms0(qh,qh,l->qn,D,c->eps);q38_rope(qh,D,c->rotary_dim,pos,c->theta);
            float *score=falloc(nsel);float mx=-INFINITY;
            int khidx=h/(QH/KVH);for(int j=0;j<nsel;j++){const float *kh=m->K[layer]+((int64_t)khidx*m->kv_cap+selected[j])*D;float a=0.f;for(int d=0;d<D;d++)a+=qh[d]*kh[d];a/=sqrtf((float)D);score[j]=a;if(a>mx)mx=a;}
            float den=0.f;for(int j=0;j<nsel;j++){score[j]=expf(score[j]-mx);den+=score[j];}
            float *oh=heads+(int64_t)s*QH*D+(int64_t)h*D;memset(oh,0,(size_t)D*sizeof(float));
            for(int j=0;j<nsel;j++){float a=score[j]/den;const float *vh=m->V[layer]+((int64_t)khidx*m->kv_cap+selected[j])*D;for(int d=0;d<D;d++)oh[d]+=a*vh[d];}
            for(int d=0;d<D;d++)oh[d]*=q38_sigmoid(qraw[D+d]);free(qh);free(score);
        }
    }
    q38_matmul(out,heads,l->o,S,QH*D,H);
    free(qp);free(kp);free(vp);free(ip);free(heads);free(qidx);free(pool);free(selected);
}

static void q38_moe(Model *m,Layer *l,int layer,const float *x,int S,float *out) {
    Cfg *c=&m->c;int H=c->hidden,E=c->experts,K=c->topk,I=c->inter,SI=c->shared_inter;
    float *logits=falloc(E),*sg=falloc(SI),*su=falloc(SI),*sh=falloc(SI),*shared=falloc(H);
    float *eg=falloc(I),*eu=falloc(I),*eh=falloc(I),*eo=falloc(H);
    for(int s=0;s<S;s++){
        const float *xs=x+(int64_t)s*H;float *ys=out+(int64_t)s*H;memset(ys,0,(size_t)H*sizeof(float));
        q38_matmul(logits,xs,l->router,1,H,E);float mx=logits[0];for(int e=1;e<E;e++)if(logits[e]>mx)mx=logits[e];
        double all=0;for(int e=0;e<E;e++){logits[e]=expf(logits[e]-mx);all+=logits[e];}
        int idx[Q38_MAX_TOPK];float val[Q38_MAX_TOPK];
        for(int z=0;z<K;z++){int best=-1;float bv=-1.f;for(int e=0;e<E;e++){int used=0;for(int j=0;j<z;j++)if(idx[j]==e)used=1;if(!used&&logits[e]>bv){bv=logits[e];best=e;}}idx[z]=rt_router_pick(best,z,E,layer);val[z]=logits[idx[z]];}
        double top=0;for(int z=0;z<K;z++)top+=val[z];double den=c->norm_topk?top:all;
        float route_gates[Q38_MAX_TOPK];
        for(int z=0;z<K;z++) route_gates[z]=(float)(val[z]/den);
        rt_route(layer,s,idx,route_gates,K); /* shared counts + post-normalization trace */
        for(int z=0;z<K;z++){
            Slot *ex=q38_expert_get(m,layer,idx[z]);q38_matmul(eg,xs,ex->gate,1,H,I);q38_matmul(eu,xs,ex->up,1,H,I);
            for(int j=0;j<I;j++)eh[j]=q38_silu(eg[j])*eu[j];q38_matmul(eo,eh,ex->down,1,I,H);
            for(int d=0;d<H;d++)ys[d]+=route_gates[z]*eo[d];
        }
        q38_matmul(sg,xs,l->sh_g,1,H,SI);q38_matmul(su,xs,l->sh_u,1,H,SI);
        for(int j=0;j<SI;j++)sh[j]=q38_silu(sg[j])*su[j];q38_matmul(shared,sh,l->sh_d,1,SI,H);
        float gate=0.f;for(int d=0;d<H;d++)gate+=xs[d]*l->sh_gate[d];gate=q38_sigmoid(gate);
        for(int d=0;d<H;d++)ys[d]+=gate*shared[d];
    }
    rt_trace_end();
    free(logits);free(sg);free(su);free(sh);free(shared);free(eg);free(eu);free(eh);free(eo);
}

static void reset_recurrent(Model *m) {
    Cfg *c=&m->c;
    for(int i=0;i<c->layers;i++)if(!c->is_attn[i]){
        memset(m->DN_rec[i],0,(size_t)c->dn_vheads*c->dn_kdim*c->dn_vdim*sizeof(float));
        memset(m->DN_conv[i],0,(size_t)c->dn_conv_dim*(c->dn_convk-1)*sizeof(float));
    }
    memset(m->PLE_conv_state,0,(size_t)c->hc_width*(c->ple_convk-1)*c->ngram_size*sizeof(float));m->ple_history_len=0;
}

static void ensure_kv(Model *m) {
    Cfg *c=&m->c;if(m->max_t<=m->kv_cap&&m->K)return;
    if(m->K){for(int i=0;i<c->layers;i++){free(m->K[i]);free(m->V[i]);free(m->IK[i]);}free(m->K);free(m->V);free(m->IK);}
    m->K=(float**)calloc((size_t)c->layers,sizeof(float*));m->V=(float**)calloc((size_t)c->layers,sizeof(float*));m->IK=(float**)calloc((size_t)c->layers,sizeof(float*));
    for(int i=0;i<c->layers;i++)if(c->is_attn[i]){
        m->K[i]=falloc((int64_t)c->kv_heads*m->max_t*c->head_dim);m->V[i]=falloc((int64_t)c->kv_heads*m->max_t*c->head_dim);m->IK[i]=falloc((int64_t)m->max_t*c->idx_dim);
    }
    m->kv_cap=m->max_t;
}

/* Run only the requested native layer interval over hyper-residual activations.
 * Segment callers supply the four-stream boundary state directly; unlike step,
 * this path deliberately does not gather embeddings or apply the final mixer. */
static void q38_layers_forward_range(Model *m,float *hyper,const int *ids,
                                     int S,int pos_base,int layer_begin,
                                     int layer_end) {
    Cfg *c=&m->c; int H=c->hidden,W=c->hc_width,C=c->hc_count;
    float *mixed=falloc((int64_t)S*H),*inject=falloc((int64_t)S*C),*block=falloc((int64_t)S*H);
    for(int i=layer_begin;i<layer_end;i++){
        Layer *l=&m->L[i];
        if(i==c->ple_layer){
            float *ple=falloc((int64_t)S*W); q38_ple(m,ids,S,hyper,ple);
            for(int64_t z=0;z<(int64_t)S*W;z++) hyper[z]+=ple[z];
            free(ple);
        }
        q38_gr_read(c,&l->attn_gr,hyper,S,mixed,inject);
        if(c->is_attn[i]) q38_attention(m,l,i,mixed,S,pos_base,block);
        else q38_deltanet(m,l,i,mixed,S,block);
        q38_gr_apply(c,hyper,block,inject,S);
        q38_gr_read(c,&l->mlp_gr,hyper,S,mixed,inject);
        q38_moe(m,l,i,mixed,S,block);
        q38_gr_apply(c,hyper,block,inject,S);
    }
    free(mixed); free(inject); free(block);
}

static float *step(Model *m,const int *ids,int S,int pos_base) {
    Cfg *c=&m->c;int H=c->hidden,W=c->hc_width,C=c->hc_count;
    float *hyper=falloc((int64_t)S*W);
    for(int s=0;s<S;s++){
        if(ids[s]<0||ids[s]>=c->vocab){fprintf(stderr,"token id %d outside vocabulary\n",ids[s]);exit(1);}
        const float *e=m->embed+(int64_t)ids[s]*H;for(int b=0;b<C;b++)memcpy(hyper+(int64_t)s*W+(int64_t)b*H,e,(size_t)H*sizeof(float));
    }
    float *mixed=falloc((int64_t)S*H),*inject=falloc((int64_t)S*C),*block=falloc((int64_t)S*H);
    for(int i=0;i<c->layers;i++){
        Layer *l=&m->L[i];
        if(i==c->ple_layer){float *ple=falloc((int64_t)S*W);q38_ple(m,ids,S,hyper,ple);for(int64_t z=0;z<(int64_t)S*W;z++)hyper[z]+=ple[z];free(ple);}
        q38_gr_read(c,&l->attn_gr,hyper,S,mixed,inject);
        if(c->is_attn[i])q38_attention(m,l,i,mixed,S,pos_base,block);else q38_deltanet(m,l,i,mixed,S,block);
        q38_gr_apply(c,hyper,block,inject,S);
        q38_gr_read(c,&l->mlp_gr,hyper,S,mixed,inject);q38_moe(m,l,i,mixed,S,block);q38_gr_apply(c,hyper,block,inject,S);
    }
    q38_gr_read(c,&m->final_gr,hyper,S,mixed,NULL);m->kv_len=pos_base+S;
    float *logit=falloc(c->vocab);q38_matmul(logit,mixed+(int64_t)(S-1)*H,m->lm_head,1,H,c->vocab);
    free(hyper);free(mixed);free(inject);free(block);return logit;
}

static void tm_report(void) {}

static void q38_layer_free(Layer *l) {
    if(!l) return;
    free(l->attn_gr.norm); free(l->attn_gr.down); free(l->attn_gr.up); free(l->attn_gr.inject);
    free(l->mlp_gr.norm); free(l->mlp_gr.down); free(l->mlp_gr.up); free(l->mlp_gr.inject);
    free(l->router); free(l->sh_g); free(l->sh_u); free(l->sh_d); free(l->sh_gate);
    free(l->q); free(l->k); free(l->v); free(l->o); free(l->qn); free(l->kn);
    free(l->idx_qk); free(l->idx_qn); free(l->idx_kn);
    free(l->dn_qkv); free(l->dn_z); free(l->dn_b); free(l->dn_a); free(l->dn_conv);
    free(l->dn_dtbias); free(l->dn_alog); free(l->dn_norm); free(l->dn_out);
    free(l->ple_key); free(l->ple_value); free(l->ple_norm_key); free(l->ple_norm_query);
    free(l->ple_norm_conv); free(l->ple_conv);
    memset(l,0,sizeof(*l));
}

static void q38_model_free(Model *m) {
    if(!m) return;
    for(int i=0;i<m->c.layers;i++) {
        q38_layer_free(&m->L[i]);
        if(m->cache) {
            /* gate/up/down share one contiguous allocation in each slot. */
            for(int s=0;s<m->cache[i].n;s++) free(m->cache[i].slots[s].gate);
            free(m->cache[i].slots); free(m->cache[i].by_expert);
        }
        free(m->DN_rec ? m->DN_rec[i] : NULL); free(m->DN_conv ? m->DN_conv[i] : NULL);
        free(m->K ? m->K[i] : NULL); free(m->V ? m->V[i] : NULL); free(m->IK ? m->IK[i] : NULL);
    }
    free(m->L); free(m->cache); free(m->DN_rec); free(m->DN_conv); free(m->K); free(m->V); free(m->IK);
    free(m->embed); free(m->lm_head);
    free(m->final_gr.norm); free(m->final_gr.down); free(m->final_gr.up); free(m->final_gr.inject);
    free(m->ple_history); free(m->PLE_conv_state); free(m->c.is_attn); st_destroy(&m->S);
    memset(m,0,sizeof(*m));
}

#endif /* COLI_QWEN38_CORE_H */
