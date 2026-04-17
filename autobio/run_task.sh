#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASK_ID=""
NUM_TRAJ=1
LOG_ROOT="$SCRIPT_DIR/logs"
DO_RENDER=1

usage() {
  cat <<'EOF'
用法:
  ./run_task.sh [选项]

选项:
  -t, --task <task_id>     指定任务名（不填则进入交互选择）
  -n, --num <N>            轨迹条数，默认 1
  -l, --logs <dir>     日志根目录，默认 ./logs
      --no-render          仅采集，不渲染视频
      --list               列出可用任务并退出
  -h, --help               显示帮助

示例:
  ./run_task.sh
  ./run_task.sh -t pipette -n 3
  ./run_task.sh -t thermal_cycler_open -n 2 --no-render
EOF
}

list_tasks() {
  python3 - <<'PY'
from run_all_tasks_once import BENCHMARK_TASK_IDS
for task in BENCHMARK_TASK_IDS:
    print(task)
PY
}

choose_task_interactive() {
  mapfile -t TASKS < <(list_tasks)
  if [ "${#TASKS[@]}" -eq 0 ]; then
    echo "未读取到任务列表。"
    exit 1
  fi
  echo "请选择要运行的任务:"
  select choice in "${TASKS[@]}"; do
    if [ -n "${choice:-}" ]; then
      TASK_ID="$choice"
      break
    fi
    echo "输入无效，请重新选择。"
  done
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -t|--task)
      TASK_ID="${2:-}"
      shift 2
      ;;
    -n|--num)
      NUM_TRAJ="${2:-}"
      shift 2
      ;;
    -l|--log-root)
      LOG_ROOT="${2:-}"
      shift 2
      ;;
    --no-render)
      DO_RENDER=0
      shift
      ;;
    --list)
      list_tasks
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1"
      usage
      exit 1
      ;;
  esac
done

if ! [[ "$NUM_TRAJ" =~ ^[0-9]+$ ]] || [ "$NUM_TRAJ" -lt 1 ]; then
  echo "--num 必须是正整数，当前: $NUM_TRAJ"
  exit 1
fi

if [ -z "$TASK_ID" ]; then
  choose_task_interactive
fi

echo "任务: $TASK_ID"
echo "轨迹条数: $NUM_TRAJ"
echo "日志目录: $LOG_ROOT"
echo "渲染视频: $([ "$DO_RENDER" -eq 1 ] && echo 是 || echo 否)"

python3 - "$TASK_ID" "$NUM_TRAJ" "$LOG_ROOT" "$DO_RENDER" <<'PY'
import sys
import traceback
from pathlib import Path

from run_all_tasks_once import _load_plugin, make_expert, render_trajectory

task_id = sys.argv[1]
num_traj = int(sys.argv[2])
log_root = Path(sys.argv[3]).resolve()
do_render = bool(int(sys.argv[4]))

_load_plugin()

task_root = log_root / task_id
task_root.mkdir(parents=True, exist_ok=True)

ok = 0
failed = 0
for i in range(num_traj):
    log_name = f"{i:03d}"
    traj_dir = task_root / log_name
    print(f"\n=== [{task_id}] 采集 {i+1}/{num_traj} -> {traj_dir} ===", flush=True)
    try:
        expert = make_expert(task_id)
        expert.reset(i)
        expert.set_serializer(log_root=task_root, log_name=log_name)
        expert.execute()
    except Exception as exc:
        failed += 1
        print(f"[ERROR] 采集失败: {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        continue

    if do_render:
        print(f"=== [{task_id}] 渲染 {i+1}/{num_traj} ===", flush=True)
        try:
            render_trajectory(traj_dir)
        except Exception as exc:
            failed += 1
            print(f"[ERROR] 渲染失败: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
            continue

    ok += 1

print("\n======== 汇总 ========")
print(f"任务: {task_id}")
print(f"成功: {ok}")
print(f"失败: {failed}")
if failed > 0:
    raise SystemExit(1)
PY
