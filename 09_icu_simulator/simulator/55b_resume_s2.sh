#!/bin/zsh
# 敏感性分析 2 从模块选择起续跑（建表与审计已于 18:26 完成；上一次会话结束时中断）
cd "${0:A:h}"
PY=../../.venv/bin/python
dir=sens_s2_v02
for step in 40_select_modules 41_system_check 42_direct_baselines 51_temporal_validation; do
  echo "$(date '+%H:%M') [$dir] $step 开始"
  env SIM_READY=v02 SIM_WORK=$dir/work SIM_OUT=$dir/outputs $PY -W ignore $step.py > $dir/$step.log 2>&1
  rc=$?
  echo "$(date '+%H:%M') [$dir] $step 结束 exit $rc"
  if [ $rc -ne 0 ]; then echo "出错"; tail -5 $dir/$step.log; exit 1; fi
done
echo "$(date '+%H:%M') 全部结束"
