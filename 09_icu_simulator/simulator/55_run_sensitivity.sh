#!/bin/zsh
# 敏感性分析 1、2（分析方案 v1.0 第 10 节）：在各自的文件夹中把完整流程重跑一遍，不触碰主分析的 work/、outputs/
# 1 = 按 anchor_year_group 划分（sens_s1_group）；2 = 用 v0.2 代替 v0.3 作为 D3 的条件（sens_s2_v02）
cd "${0:A:h}"
PY=../../.venv/bin/python
run_variant () {
  local dir=$1; shift
  for step in 21_build_training_v1 22_data_audit 40_select_modules 41_system_check 42_direct_baselines 51_temporal_validation; do
    echo "$(date '+%H:%M') [$dir] $step 开始"
    env "$@" SIM_WORK=$dir/work SIM_OUT=$dir/outputs $PY -W ignore $step.py > $dir/$step.log 2>&1
    rc=$?
    echo "$(date '+%H:%M') [$dir] $step 结束 exit $rc"
    if [ $rc -ne 0 ]; then echo "[$dir] 出错，停止该方案"; tail -5 $dir/$step.log; return 1; fi
  done
}
run_variant sens_s1_group SIM_SPLIT=group
run_variant sens_s2_v02 SIM_READY=v02
echo "$(date '+%H:%M') 全部结束"
