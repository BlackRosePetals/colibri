#!/bin/bash
# ============================================================
# benchmark_run.sh — fmt=2 (per-row int4) M1 Ultra benchmark matrix
#
# Logs -> docs/fmt2/<NN>_<name>.log
# Summary -> docs/fmt2/SUMMARY.md
#
# Incremental: a run is skipped if its log exists, completed, and was
# recorded with MTP=0 ("[MTP] absent"). Stale/invalid logs (incomplete or
# MTP-active) are quarantined to docs/fmt2/invalid/ and re-run.
#
# .coli_usage is snapshotted to docs/fmt2/colibri_usage.snapshot and
# restored before every run, so all runs pin from identical expert history
# (the engine appends to .coli_usage during decode, which otherwise grows
# the pin set run-over-run). REFRESH_SNAP=1 forces a re-snapshot.
#
# Usage:
#   ./benchmark_run.sh              # full matrix (skips valid logs)
#   ./benchmark_run.sh E_optimal    # single run
#   ./benchmark_run.sh ram          # all RAM-sweep runs
#   FORCE=1 ./benchmark_run.sh      # archive ALL existing logs to repeats/<ts>/ and re-run
#   DRY_RUN=1 ./benchmark_run.sh    # print decisions/commands, touch nothing
#   CAP=36 ./benchmark_run.sh ram120 # override expert cache cap (default 33)
# ============================================================
set -u
cd "$(dirname "$0")"                       # c/

MODEL=/Users/joe/mlx-models/GLM-5.2-colibri-int4-perrow
PROMPT="Compare the myths of Lucifer and Prometheus"
LOGDIR=.
COLIPATH=../../c/coli
NGEN=1024
# Expert cache slots/layer -> --cap. v1.4.0 detects this SSD as "fast" and
# defaults to cap=1 (page-cache favored): measured 0.98 vs 1.45 tok/s on this
# machine (warmup, 2026-08-04). 33 reproduces the pre-rebase auto-raised value
# at ram 110; applied flat at every ram level (old run scaled 31-38 there).
CAP=${CAP:-33}

mkdir -p "$LOGDIR"

# ---- subset selection ----
WANT=()
[ $# -gt 0 ] && WANT=("$@")
want() {
  [ ${#WANT[@]} -eq 0 ] && return 0
  local w; for w in "${WANT[@]}"; do
    case "$1" in *"$w"*) return 0;; esac
  done
  return 1
}

# ---- preflight ----
[ -x ${COLIPATH} ] || { echo "ERROR: ${COLIPATH} not built (make METAL=1)"; exit 1; }
[ -d "$MODEL" ] || { echo "ERROR: model dir missing: $MODEL"; exit 1; }

WIRED=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
[ "$WIRED" = "0" ] && echo "WARNING: iogpu.wired_limit_mb=0 — run: sudo sysctl iogpu.wired_limit_mb=120832"

USAGE="$MODEL/.coli_usage"
USAGE_SNAP="$LOGDIR/colibri_usage.snapshot"   # frozen expert history for the whole matrix
INVALID_DIR="$LOGDIR/invalid"                 # quarantine for incomplete/foreign logs
MIN_USAGE_SELECTIONS=1000000                  # maturity guard (measured mature file: 6.4M)
ARCHIVE_DIR="$LOGDIR/repeats/$(date -u +%Y%m%dT%H%M%SZ)"  # FORCE=1 destination for old logs

if [ -f "$USAGE" ]; then
  USAGE_N=$(wc -l < "$USAGE")
  echo "[usage] $USAGE_N entries in live .coli_usage"
else
  echo "WARNING: no .coli_usage — cold pin"
  USAGE_N=0
fi

# ---- snapshot: create once (or on REFRESH_SNAP=1), reuse thereafter ----
if [ -n "${REFRESH_SNAP:-}" ] || [ ! -f "$USAGE_SNAP" ]; then
  if [ -f "$USAGE" ]; then
    if [ -z "${DRY_RUN:-}" ]; then
      cp "$USAGE" "$USAGE_SNAP"
      echo "[usage] snapshot refreshed -> $USAGE_SNAP"
    else
      echo "[usage] DRY_RUN: would snapshot $USAGE -> $USAGE_SNAP"
    fi
  fi
fi
if [ -f "$USAGE_SNAP" ]; then
  SNAP_SEL=$(awk '{s+=$3} END{print s+0}' "$USAGE_SNAP")
  echo "[usage] snapshot: $(wc -l < "$USAGE_SNAP") entries, $SNAP_SEL selections"
  if [ "$SNAP_SEL" -lt "$MIN_USAGE_SELECTIONS" ]; then
    echo "WARNING: usage snapshot looks immature ($SNAP_SEL < $MIN_USAGE_SELECTIONS selections) — pin set may be unstable"
  fi
else
  echo "WARNING: no usage snapshot — runs will pin from live (drifting) history"
fi

# ---- environment snapshot ----
if [ -z "${DRY_RUN:-}" ]; then
  {
    echo "# benchmark_run environment snapshot"
    date -u +"# date: %Y-%m-%dT%H:%M:%SZ"
    echo "# wired_limit_mb: $WIRED"
    echo "# usage_lines: $USAGE_N"
    echo "# usage_snapshot: $USAGE_SNAP (${SNAP_SEL:-0} selections)"
    sw_vers | sed 's/^/# /'
    echo "# model: $MODEL"
    df -h /Users/joe 2>/dev/null | sed 's/^/# /'
  } > "$LOGDIR/00_environment.txt"
fi

echo "[setup] logs -> $LOGDIR"

# ---- run helper ----
# All runs are MTP=0 (see phase-4 note below for why). A log is only valid if
# it is complete (has a final decode line) and was recorded with MTP absent.
run() {
  local id="$1" ngen="$2" ram="$3"; shift 3
  want "$id" || return 0
  local log="$LOGDIR/${id}.log"
  if [ -f "$log" ]; then
    if [ -n "${FORCE:-}" ]; then
      echo "[FORCE] $id: archiving existing log -> $ARCHIVE_DIR/"
      if [ -z "${DRY_RUN:-}" ]; then
        mkdir -p "$ARCHIVE_DIR"
        mv "$log" "$ARCHIVE_DIR/"
      fi
    elif grep -q '^\[MTP] absent' "$log" && grep -q 'decode .* tok/s' "$log"; then
      echo "[SKIP] $id: valid log exists ($log)"
      return 0
    else
      echo "[INVALID] $id: quarantining -> $INVALID_DIR/"
      if [ -z "${DRY_RUN:-}" ]; then
        mkdir -p "$INVALID_DIR"
        mv "$log" "$INVALID_DIR/"
      fi
    fi
  fi
  local cmd=(env COLI_METAL=1 DIRECT=1 MTP=0 "$@" ${COLIPATH} run \
             --model "$MODEL" --ram "$ram" --cap "$CAP" --ngen "$ngen" "$PROMPT")
  echo "[RUN] $id: ${cmd[*]}"
  if [ -n "${DRY_RUN:-}" ]; then return; fi
  # freeze expert history: every run pins from the same snapshot
  if [ -f "$USAGE_SNAP" ]; then
    cp "$USAGE_SNAP" "$USAGE"
    echo "[usage] $id: snapshot restored (md5 $(md5 -q "$USAGE"))"
  fi
  {
    echo "# run: $id"
    echo "# date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# cmd: ${cmd[*]}"
    echo "# usage_md5: $(md5 -q "$USAGE" 2>/dev/null || echo none)"
    SECONDS=0
    "${cmd[@]}" 2>&1
    echo "# wall: ${SECONDS}s"
  } | tee "$log"
}

# ============================================================
# Run matrix
# ============================================================

# Warm-up (128 tokens) — warms cache, not timed
echo "--- phase 0: warm-up (not benchmarked) ---"
run 00_warmup_128 128 110 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8

echo "--- phase 1: A-E configs (match M5 report) ---"
# Note: A and B are identical in the fmt=2 plan (both default flags).
# We merge them as '01_Abaseline' to save a 13-minute duplicate.
# MTP=0 is baked into the run helper (see comment there).
run 01_Abaseline   $NGEN 110
run 03_Cpipe_only  $NGEN 110 PIPE=1 PIPE_WORKERS=8
run 04_Dno_omp     $NGEN 110 COLI_NO_OMP_TUNE=1
run 05_Eoptimal    $NGEN 110 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8

echo "--- phase 2: M1 Ultra scaling ---"
run 06_Epipe12     $NGEN 110 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=12
run 07_Epipe16     $NGEN 110 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=16

echo "--- phase 3: memory elbow (find compression knee) ---"
run 08_ram105      $NGEN 105 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8
run 09_ram112      $NGEN 112 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8
run 10_ram115      $NGEN 115 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8
run 11_ram118      $NGEN 118 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8
run 12_ram120      $NGEN 120 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8
# elbow probe: MTP=0 RSS tracked ~12 GB below projection up to ram 120 (106.4 GB
# actual) with no thrash; ram 125 should land ~108-109 GB — near the historical
# ~109.6+ GB thrash zone. Runs LAST: if tok/s collapses (watch the log), kill it
# — the rest of the matrix is already complete.
run 13_ram125      $NGEN 125 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8

# --- phase 4 (MTP) dropped ---
# MTP=1 is a strict loss on this machine: acceptance was real (63-74%,
# 1.63-1.74 tok/fw — accidental MTP-active runs of 2026-08-03) but the
# +12-15 GB RSS pushes past the memory knee into swap thrash (0.07-1.04 tok/s
# vs 1.47 for MTP=0). No MTP run in the matrix.

# ============================================================
# Summary
# ============================================================
# leave the live .coli_usage exactly as the frozen snapshot (benchmark
# selections are prompt-specific and would otherwise pollute organic history)
if [ -z "${DRY_RUN:-}" ] && [ -f "$USAGE_SNAP" ] && [ -f "$USAGE" ]; then
  cp "$USAGE_SNAP" "$USAGE"
  echo "[usage] live .coli_usage restored from snapshot"
fi

if [ -z "${DRY_RUN:-}" ]; then
  echo "[summary] generating $LOGDIR/SUMMARY.md ..."
  {
    echo "# fmt=2 benchmark summary"
    echo "generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo ""
    echo "| run | tok/s | decode s | hit % | disk wait s | matmul s | attn s | attn gpu-sched s | expert gpu-sched s | RSS GB |"
    echo "|---|---|---|---|---|---|---|---|---|---|---|"
    for f in "$LOGDIR"/[0-9]*.log; do
      [ -f "$f" ] || continue
      id=$(basename "$f" .log)
      # extract final decode line (prefill...| decode...| RSS)
      line=$(grep -E 'tok/s.*hit rate.*RSS' "$f" | tail -1)
      prof=$(grep '^PROFILE:' "$f" | tail -1)
      mattn=$(grep '^METAL-ATTN:' "$f" | tail -1)
      mmet=$(grep '^METAL: blocchi' "$f" | tail -1)

      toks="-" dec="-" hit="-" rss="-" dw="-" mm="-" att="-" ags="-" egs="-"
      if [ -n "$line" ]; then
        toks=$(echo "$line" | sed -n 's/.*decode [0-9]* tokens in \([0-9.]*\)s (\([0-9.]*\) tok\/s).*/\2/p')
        dec=$(echo "$line" | sed -n 's/.*decode [0-9]* tokens in \([0-9.]*\)s .*/\1/p')
        hit=$(echo "$line" | sed -n 's/.*expert hit rate \([0-9.]*\)%.*/\1/p')
        rss=$(echo "$line" | sed -n 's/.*RSS \([0-9.]*\) GB.*/\1/p')
      fi
      if [ -n "$prof" ]; then
        # try new "service / Ns wait" format first
        dw=$(echo "$prof" | sed -n 's/.*expert-disk [0-9.]*s service \/ \([0-9.]*\)s wait.*/\1/p')
        # fall back to old "expert-disk Ns" (no wait split)
        [ -z "$dw" ] && dw=$(echo "$prof" | sed -n 's/.*expert-disk \([0-9.]*\)s.*/\1/p')
        mm=$(echo "$prof" | sed -n 's/.*expert-matmul \([0-9.]*\)s.*/\1/p')
        att=$(echo "$prof" | sed -n 's/.*attention \([0-9.]*\)s.*/\1/p')
      fi
      # gpu-sched canary: scheduling stall = memory-pressure thrash
      # (kernel time stays flat; sched/wait explodes when macOS swaps)
      [ -n "$mattn" ] && ags=$(echo "$mattn" | sed -n 's/.*gpu-sched \([0-9.]*\)s.*/\1/p')
      # METAL line has no sched split; expert sched ~= gpu-wall - kernel
      if [ -n "$mmet" ]; then
        egs=$(echo "$mmet" | sed -n 's/.*gpu-wall \([0-9.]*\)s (kernel \([0-9.]*\)s).*/\1 \2/p' \
              | awk '{if (NF==2) printf "%.1f", $1-$2}')
        [ -z "$egs" ] && egs="-"
      fi
      echo "| $id | ${toks:- -} | ${dec:- -} | ${hit:- -} | ${dw:- -} | ${mm:- -} | ${att:- -} | ${ags:- -} | ${egs:- -} | ${rss:- -} |"
    done
    echo ""
    echo "## environment"
    cat "$LOGDIR/00_environment.txt"
  } > "$LOGDIR/SUMMARY.md"
  echo "[summary] $LOGDIR/SUMMARY.md ready"
fi

echo "[done] all runs complete"
