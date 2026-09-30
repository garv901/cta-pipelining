#!/bin/bash
# usage: pick_gpus.sh [N=2] [MAX_MIN=120]; prints N comma-separated physical GPU ids with no compute apps,
# stable over 3 polls 20 s apart; prefers same-NUMA pairs (3,2) (2,3) (1,0) (0,1). Exit 1 on timeout.
N=${1:-2}; MAX=${2:-120}
free() {
  local busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read i u; do
    grep -q "$u" <<< "$busy" || echo $i; done | sort | tr '\n' ' '
}
pick() {
  local f=" $1 "
  if [ "$N" = 2 ]; then for p in "3 2" "2 3" "1 0" "0 1" "3 1" "3 0" "2 1" "2 0" "1 3" "0 3" "1 2" "0 2"; do
    set -- $p; [[ "$f" == *" $1 "* && "$f" == *" $2 "* ]] && { echo "$1,$2"; return; }; done
  else set -- $1; [ $# -ge "$N" ] && echo "${@:1:$N}" | tr ' ' ','; fi
}
end=$(( $(date +%s) + MAX*60 ))
while [ $(date +%s) -lt $end ]; do
  a=$(pick "$(free)"); ok=1
  if [ -n "$a" ]; then for k in 1 2; do sleep 20; [ "$(pick "$(free)")" = "$a" ] || ok=0; done
    [ $ok = 1 ] && { echo "$a"; exit 0; }; else sleep 30; fi
done; echo timeout >&2; exit 1
