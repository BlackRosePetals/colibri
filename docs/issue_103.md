[Performance]: tok/s of 2.06 tok/s !!! · Issue #103 · JustVugg/colibri · GitHub

# [Performance]: tok/s of 2.06 tok/s !!! #103

**RDouglasSharp** opened on Jul 12, 2026

## Commit
pr-72-metal

## Hardware and storage
Apple M5 Max 128Mb 14" 1
8 core (6 Super and 12 Performance)
40 Core Metal 4 GPU

## Software environment
sudo sysctl iogpu.wired_limit_mb=120832

## Build and benchmark commands
COLI_METAL=1 DIRECT=1 MTP=0 ./coli run --model /Users/doug/glm52_i4 "Compare the myths of Lucifer and Prometheus" --ram 110

## Results
```
COLI_METAL=1 DIRECT=1 MTP=0 ./coli run --model /Users/doug/glm52_i4 "Compare the myths of Lucifer and Prometheus" --ram 110

     ▄▀▀▀▄  ▄        colibrì v1.0
  ▄▄▄▄▀▀▀▀▄▀▀        piccolo motore, modello immenso
      ▀▀▀▀▀▀▀        GLM-5.2 · 744B MoE · int4 · streaming CPU
        ▀▀▀▀         run
          ▀          
  ──────────────────────────────────────────────────────────
[METAL] mode: batched routed experts on GPU (unified-memory zero-copy)
== Motore C GLM (glm_moe_dsa), cache=8 expert/layer | expert@8-bit densa@8-bit | idot: neon ==
caricato in 0.81s | densa residente: 9647.10 MB | layers=78 experts=256 | MTP assente (draft=0)
[MTP] assente (draft=0)
[USAGE] storia expert: 2936400 selezioni (/Users/doug/glm52_i4/.coli_usage)
[PIN] hot-store: 2478 expert in RAM (46.9 GB) in 3s da /Users/doug/glm52_i4/.coli_usage
[PIN] mlock: 46.9 GB inchiodati in RAM fisica / wired in physical RAM (niente compressione/no compression) in 1s
[RAM_GB=110.0] cap ALZATO 8->33: il budget lo consente (proiezione picco 109.9 GB; CAP_RAISE=0 per disattivare)
[stop] 3 token di stop: 154820 154827 154829
prompt: 13 token | genero fino a 1024 (stop EOS=154820) | draft n-gram=0
[gMASK]<sop><|user|>Compare the myths of Lucifer and Prometheus<|assistant|></think>[prefill] layer 1/78 · 13 token
[prefill] layer 5/78 · 13 token
[prefill] layer 9/78 · 13 token
[prefill] layer 13/78 · 13 token
[prefill] layer 17/78 · 13 token
[prefill] layer 21/78 · 13 token
[prefill] layer 25/78 · 13 token
[prefill] layer 29/78 · 13 token
[prefill] layer 33/78 · 13 token
[prefill] layer 37/78 · 13 token
[prefill] layer 41/78 · 13 token
[prefill] layer 45/78 · 13 token
[prefill] layer 49/78 · 13 token
[prefill] layer 53/78 · 13 token
[prefill] layer 57/78 · 13 token
[prefill] layer 61/78 · 13 token
[prefill] layer 65/78 · 13 token
[prefill] layer 69/78 · 13 token
[prefill] layer 73/78 · 13 token
[prefill] layer 77/78 · 13 token
[prefill] layer 78/78 · 13 token
The myths of Lucifer (in Christian tradition) and Prometheus (in Greek tradition)
[t=16  RSS 97.83 GB  hit 61%  1.23 tok/s  1.00 tok/fw]  represent two of the most profound and complex archetypes in human mythology: the "
[t=32  RSS 97.89 GB  hit 67%  1.60 tok/s  1.00 tok/fw]  rebel" or the "trickster" who defies the heavens.
[t=48  RSS 97.89 GB  hit 70%  1.81 tok/s  1.00 tok/fw]  Though they originate from entirely different cultural and theological contexts, their narratives share striking structural
[t=64  RSS 97.89 GB  hit 70%  1.85 tok/s  1.00 tok/fw]  and thematic similarities. as well as distinct differences.

Here is a detailed comparison of
[t=80  RSS 97.89 GB  hit 71%  1.90 tok/s  1.00 tok/fw]  the two figures:

## **1. Core Identity and Sphere of Influence**

**
[t=96  RSS 97.89 GB  hit 72%  1.96 tok/s  1.00 tok/fw]  Lucifer**
*   **Origin:**** The name "Lucifer" translates
[t=112  RSS 97.89 GB  hit 73%  2.02 tok/s  1.00 tok/fw]  from Latin *lux* (light) + *ferre* (to carry
[t=128  RSS 97.89 GB  hit 73%  2.05 tok/s  1.00 tok/fw]  or to bring), meaning "Light-Bringer" or "Morning Star."
[t=144  RSS 97.89 GB  hit 74%  2.08 tok/s  1.00 tok/fw]  The title originally referred to the planet Venus in its morning appearance, signifying the
[t=160  RSS 97.89 GB  hit 74%  2.09 tok/s  1.00 tok/fw]  dawn and bringing of light to dispel the darkness of night.
*   **
[t=176  RSS 97.89 GB  hit 74%  2.10 tok/s  1.00 tok/fw]  Christian Tradition:**** In Christian theology, Lucifer is often identified with Satan, the
[t=192  RSS 97.89 GB  hit 74%  2.11 tok/s  1.00 tok/fw]  adversary. However, strictly speaking, Lucifer is a Latin translation of the Hebrew phrase
[t=208  RSS 97.89 GB  hit 74%  2.12 tok/s  1.00 tok/fw]  *helel ben-shahar* (meaning "Morning Star, son
[t=224  RSS 97.89 GB  hit 74%  2.13 tok/s  1.00 tok/fw]  of the Dawn") found in Isaiah 14:12. This passage was a
[t=240  RSS 97.89 GB  hit 74%  2.13 tok/s  1.00 tok/fw]  prophecy against the King of Babylon, mocking his human arrogance and ambition.
*  
[t=256  RSS 97.89 GB  hit 74%  2.12 tok/s  1.00 tok/fw]  **Mythological Evolution:**** Over time, through the influence of literature like
[t=272  RSS 97.89 GB  hit 74%  2.13 tok/s  1.00 tok/fw]  John Milton's *Paradise Lost* (1667) and Dante's
[t=288  RSS 97.89 GB  hit 74%  2.13 tok/s  1.00 tok/fw]  *Inferno* (1320), Lucifer became firmly established in the popular
[t=304  RSS 97.89 GB  hit 74%  2.12 tok/s  1.00 tok/fw]  consciousness as the name of the archangel who, driven by pride, rebelled
[t=320  RSS 97.89 GB  hit 74%  2.12 tok/s  1.00 tok/fw]  against God. He was cast down to hell...
*   **Role:****
[t=336  RSS 97.89 GB  hit 74%  2.12 tok/s  1.00 tok/fw]  In Christian theology, Lucifer/Satan plays the role of the "adversary
[t=352  RSS 97.89 GB  hit 74%  2.12 tok/s  1.00 tok/fw]  " or the "accuser." He is the great antagonist in the Christian cosmic
[t=368  RSS 97.89 GB  hit 74%  2.13 tok/s  1.00 tok/fw]  drama: a fallen angel who defies the divine order and brings evil and suffering
[t=384  RSS 97.89 GB  hit 74%  2.13 tok/s  1.00 tok/fw]  into the world through his deception of Adam and Eve in the Garden of Eden.
[t=400  RSS 97.89 GB  hit 74%  2.12 tok/s  1.00 tok/fw]  His "crime" is primarily one of *hubris*—overweening
[t=416  RSS 97.89 GB  hit 74%  2.12 tok/s  1.00 tok/fw]  ambition and pride—which leads to a rebellion against the divine hierarchy.
*   **
[t=432  RSS 97.89 GB  hit 73%  2.12 tok/s  1.00 tok/fw]  Symbolic Meaning:**** Symbolically, Lucifer represents the sin of pride ("*
[t=448  RSS 97.89 GB  hit 73%  2.11 tok/s  1.00 tok/fw]  invidia*" in Latin, meaning "envy"). He is the embodiment of
[t=464  RSS 97.89 GB  hit 73%  2.11 tok/s  1.00 tok/fw]  the rebellious intellect who seeks to become equal to the gods..

**Prom
[t=480  RSS 97.89 GB  hit 73%  2.11 tok/s  1.00 tok/fw]  ometheus**
*   **Origin:**** In Greek myth, Prometheus ("Fores
[t=496  RSS 97.89 GB  hit 73%  2.11 tok/s  1.00 tok/fw] ightful") is a second-generation Titan, a race of powerful beings who ruled
[t=512  RSS 97.89 GB  hit 73%  2.11 tok/s  1.00 tok/fw]  the cosmos during the Golden Age.* This is a key difference: while Lucifer
[t=528  RSS 97.89 GB  hit 73%  2.11 tok/s  1.00 tok/fw]  is an angel (a servant of God, created solely to obey and worship),
[t=544  RSS 97.89 GB  hit 73%  2.11 tok/s  1.00 tok/fw]  Prometheus is a Titan, a primordial cosmic entity, born of the Earth (
[t=560  RSS 97.89 GB  hit 73%  2.11 tok/s  1.00 tok/fw]  Gaia) and Sky (Uranus).
*   **Mythological
[t=576  RSS 97.89 GB  hit 73%  2.11 tok/s  1.00 tok/fw]  Role:**** Prometheus is the quintessential "trickster" and culture-hero
[t=592  RSS 97.89 GB  hit 73%  2.11 tok/s  1.00 tok/fw]  of Greek myth. He is the creator of humanity, fashioning the first humans
[t=608  RSS 97.89 GB  hit 73%  2.11 tok/s  1.00 tok/fw]  from clay. He is the benefactor of humanity, whose profound love for his
[t=624  RSS 97.89 GB  hit 73%  2.11 tok/s  1.00 tok/fw]  human creations drives him to steal fire from the gods to give to humans, enabling
[t=640  RSS 97.89 GB  hit 73%  2.11 tok/s  1.00 tok/fw]  human progress and the dawn of civilization. Because of this theft, he is chained
[t=656  RSS 97.89 GB  hit 73%  2.11 tok/s  1.00 tok/fw]  to a rock in the Caucasus Mountains, where an eagle (the symbol of
[t=672  RSS 97.89 GB  hit 73%  2.11 tok/s  1.00 tok/fw]  Zeus) eats his liver daily in perpetuity.
*   **Symbolic Meaning
[t=688  RSS 97.89 GB  hit 73%  2.11 tok/s  1.00 tok/fw]  :**** Prometheus represents human aspiration and intellect. He embodies "Promethean"
[t=704  RSS 97.89 GB  hit 73%  2.10 tok/s  1.00 tok/fw]  ambition—the drive to achieve greatness through technological and intellectual mastery over nature. He is
[t=720  RSS 97.89 GB  hit 73%  2.10 tok/s  1.00 tok/fw]  the patron of human striving and civilization. His myth explores the relationship between *mind
[t=736  RSS 97.89 GB  hit 73%  2.10 tok/s  1.00 tok/fw]  * (human intellect and craft) and *matter* (the physical world).


[t=752  RSS 97.89 GB  hit 73%  2.11 tok/s  1.00 tok/fw]  ## **2. Motivations: Rebellion, Pride, and Human Agency**


[t=768  RSS 97.89 GB  hit 73%  2.11 tok/s  1.00 tok/fw]  **Lucifer: Rebellion as Sin**
In the Christian tradition, Lucifer's rebellion
[t=784  RSS 97.89 GB  hit 73%  2.10 tok/s  1.00 tok/fw]  is a sin born of pride and ambition. In Christian theology, angels are strictly
[t=800  RSS 97.89 GB  hit 73%  2.10 tok/s  1.00 tok/fw]  created beings, whose purpose is to serve and praise God.* When Lucifer says
[t=816  RSS 97.89 GB  hit 73%  2.09 tok/s  1.00 tok/fw]  , "I will ascend into heaven... I will make myself like the Most High
[t=832  RSS 97.89 GB  hit 73%  2.09 tok/s  1.00 tok/fw]  " (Isaiah :13-14), he is committing the sin
[t=848  RSS 97.89 GB  hit 73%  2.09 tok/s  1.00 tok/fw]  of *hubris* (overwhelming ambition) and *invidia* (
[t=864  RSS 97.89 GB  hit 73%  2.09 tok/s  1.00 tok/fw]  envy). He refuses to accept his divinely ordained place in the cosmos.

[t=880  RSS 97.89 GB  hit 73%  2.09 tok/s  1.00 tok/fw]  **Prometheus: Rebellion as Human Agency**
In contrast, Prometheus's defiance of
[t=896  RSS 97.89 GB  hit 73%  2.08 tok/s  1.00 tok/fw]  Zeus is not a sin, but an act of profound compassion and human agency.
[t=912  RSS 97.89 GB  hit 73%  2.08 tok/s  1.00 tok/fw]  His "crime" was motivated by pity for human beings' physical weakness and ignorance
[t=928  RSS 97.89 GB  hit 73%  2.08 tok/s  1.00 tok/fw]  . Unlike Lucifer, who is jealous of God, Prometheus is driven by *ag
[t=944  RSS 97.89 GB  hit 73%  2.07 tok/s  1.00 tok/fw]  ape* (unconditional love) for his creations, humans. By stealing fire
[t=960  RSS 97.89 GB  hit 73%  2.07 tok/s  1.00 tok/fw]  , he acts as a culture-bringer, giving humans the tools of civilization—
[t=976  RSS 97.89 GB  hit 73%  2.07 tok/s  1.00 tok/fw]  language, science,, and the arts—enabling them to become masters of
[t=992  RSS 97.89 GB  hit 73%  2.07 tok/s  1.00 tok/fw]  their own fate and to progress.
**Lucifer and Prometheus as "Light-
[t=1008  RSS 97.89 GB  hit 73%  2.06 tok/s  1.00 tok/fw]  Bringers"**
Despite their different roles, both Lucifer and Prometheus are "light

---
1024 token in 496.34s (2.06 tok/s) | hit-rate expert 72.5% | RSS 97.89 GB
expert caricati/token: 607.0 (per-layer 8.09 su 75; baseline topk=8) | TOPK=0 TOPP=0.00
speculazione: 1.00 token/forward (1023 fw per 1024 tok) | MTP acceptance 0% (0/0)
PROFILO: expert-disk 266.259s | expert-matmul 109.471s | attention 100.133s (di cui kvb 0.038s) | lm_head 0.004s | altro 20.475s
METAL-ATTN: layer GPU 79794 | gpu-wall 96.35s (kernel 76.36s | cpu-sched 0.62s gpu-sched 11.63s)
METAL: blocchi GPU 142391 | fallback CPU 0 | expert su GPU 618242 | setup 15.35s gpu-wall 86.02s (kernel 35.24s) scatter 1.46s
```

## Baseline comparison

Note that this Metal PR is based on an older base, missing features like interleaved expert loading.
So once properly merged, the tok/sec may be slightly higher!
