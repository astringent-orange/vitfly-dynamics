#!/bin/bash

# Pass number of rollouts as argument
if [ $1 ]
then
  N="$1"
else
  N=5
fi

echo $2

state_human=0
random_env=0
fixed_env=0
force_rviz=0
benchmark_mode="${VITFLY_BENCHMARK_MODE:-0}"
ROS_PID=""
PY_PID=""
COMP_PID=""
cleanup_in_progress=0
env_count="${VITFLY_ENV_COUNT:-101}"
env_level="${VITFLY_ENV_LEVEL:-dynamic_astar_medium}"
des_vel="${VITFLY_DES_VEL:-4.0}"
offset="${VITFLY_OFFSET:-0}"
model_path="${VITFLY_MODEL_PATH:-}"
phase_seed_override="${VITFLY_DYNAMIC_PHASE_SEED:-}"
phase_seed_base="${VITFLY_DYNAMIC_PHASE_SEED_BASE:-1000}"
policy_config="${VITFLY_POLICY_CONFIG:-}"
case_config="${VITFLY_CASE_CONFIG:-}"
evaluation_profile="${VITFLY_EVALUATION_PROFILE:-strict}"
result_path="${VITFLY_EVALUATION_PATH:-evaluation.yaml}"
real_time_factor="${VITFLY_REAL_TIME_FACTOR:-}"
reuse_simulator="${VITFLY_REUSE_SIMULATOR:-0}"
simulator_session_id="${VITFLY_SIMULATOR_SESSION_ID:-}"
python_bin="${VITFLY_PYTHON:-python3}"
planner_adapter="${VITFLY_PLANNER_ADAPTER:-}"
planner_launch="${VITFLY_PLANNER_LAUNCH:-}"
planner_ready_topic="${VITFLY_PLANNER_READY_TOPIC:-}"

for arg in "${@:3}"
do
  if [ "$arg" = "human" ]
  then
    state_human=1
  elif [ "$arg" = "random_env" ]
  then
    random_env=1
  elif [ "$arg" = "fixed_env" ]
  then
    fixed_env=1
  elif [ "$arg" = "rviz" ] || [ "$arg" = "debug" ]
  then
    force_rviz=1
  elif [[ "$arg" == env_count=* ]]
  then
    env_count="${arg#env_count=}"
  elif [[ "$arg" == env_level=* ]]
  then
    env_level="${arg#env_level=}"
  elif [[ "$arg" == phase_seed=* ]]
  then
    phase_seed_override="${arg#phase_seed=}"
  elif [[ "$arg" == phase_seed_base=* ]]
  then
    phase_seed_base="${arg#phase_seed_base=}"
  elif [[ "$arg" == offset=* ]]
  then
    offset="${arg#offset=}"
  elif [[ "$arg" == model_path=* ]]
  then
    model_path="${arg#model_path=}"
  elif [[ "$arg" == policy_config=* ]]
  then
    policy_config="${arg#policy_config=}"
  elif [[ "$arg" == case_config=* ]]
  then
    case_config="${arg#case_config=}"
  elif [[ "$arg" == evaluation_profile=* ]]
  then
    evaluation_profile="${arg#evaluation_profile=}"
  elif [[ "$arg" == result_path=* ]]
  then
    result_path="${arg#result_path=}"
  elif [[ "$arg" == real_time_factor=* ]]
  then
    real_time_factor="${arg#real_time_factor=}"
  elif [ "$arg" = "reuse_simulator" ] || [ "$arg" = "external_simulator" ]
  then
    reuse_simulator=1
  elif [ "$arg" = "benchmark_mode" ]
  then
    benchmark_mode=1
    export VITFLY_BENCHMARK_MODE=1
  elif [[ "$arg" == model_type=* || "$arg" == frame_offset=* ]]
  then
    echo "[LAUNCH SCRIPT] model_type/frame_offset are obsolete; use offset=0,1,2"
    exit 1
  fi
done

if [ "$benchmark_mode" = "1" ]
then
  export VITFLY_BENCHMARK_MODE=1
  if [ "$N" != "1" ]
  then
    echo "[LAUNCH SCRIPT] benchmark_mode requires exactly one rollout per launch."
    exit 3
  fi
fi

if ! [[ "$offset" =~ ^[012]$ ]]
then
  echo "[LAUNCH SCRIPT] offset must be 0, 1, or 2, got: $offset"
  exit 1
fi

export VITFLY_POLICY_CONFIG="$policy_config"
export VITFLY_CASE_CONFIG="$case_config"
export VITFLY_EVALUATION_PROFILE="$evaluation_profile"
export VITFLY_EVALUATION_PATH="$result_path"

if [ -n "$planner_adapter" ] && [ "$benchmark_mode" != "1" ]; then
  echo "[LAUNCH SCRIPT] planner adapters are only supported in benchmark mode."
  exit 1
fi
if [ -n "$planner_adapter" ] && [ -z "$planner_launch" ]; then
  echo "[LAUNCH SCRIPT] planner adapter requires VITFLY_PLANNER_LAUNCH."
  exit 3
fi

if ! [[ "$env_count" =~ ^[1-9][0-9]*$ ]]
then
  echo "[LAUNCH SCRIPT] env_count must be a positive integer, got: $env_count"
  exit 1
fi

if ! [[ "$phase_seed_base" =~ ^[0-9]+$ ]]
then
  echo "[LAUNCH SCRIPT] phase_seed_base must be a nonnegative integer, got: $phase_seed_base"
  exit 1
fi

if [ "$2" = "vision" ]
then
  echo
  echo "[LAUNCH SCRIPT] Vision based!"
  echo
  run_competition_args="--vision_based"
  if [ -n "$real_time_factor" ]
  then
    realtimefactor="real_time_factor:=$real_time_factor"
  else
    realtimefactor=""
  fi
  rviz_enabled=True
elif [ "$2" = "state" ]
then
  echo
  echo "[LAUNCH SCRIPT] State based!"
  echo
  run_competition_args="--state_based"
  rviz_enabled=False
  if ((state_human))
  then
    run_competition_args="--keyboard"
    realtimefactor="real_time_factor:=1.0"
    rviz_enabled=True
  else
    run_competition_args=""
    realtimefactor="real_time_factor:=10.0"
    if ((!fixed_env))
    then
      random_env=1
    fi
  fi
else
  echo
  echo "[LAUNCH SCRIPT] Unknown or empty second argument: $2, only 'vision' or 'state' allowed!"
  echo
  exit 1
fi

if ((force_rviz))
then
  rviz_enabled=True
fi

publish_rgb=True
publish_optical_flow=True

# Batch benchmarks are headless by design.  Apply this after the optional
# rviz/debug override so no benchmark invocation can accidentally launch RViz
# or publish image modalities that the policy does not consume.
if [ "$benchmark_mode" = "1" ]
then
  rviz_enabled=False
  publish_rgb=False
  publish_optical_flow=False
fi

if [ "$reuse_simulator" = "1" ] && [ "$benchmark_mode" != "1" ]
then
  echo "[LAUNCH SCRIPT] reuse_simulator is only valid in benchmark mode."
  exit 1
fi

# Set Flightmare Path if it is not set
if [ -z $FLIGHTMARE_PATH ]
then
  export FLIGHTMARE_PATH=$PWD/flightmare
fi

ros_master_ready() {
  rostopic list >/dev/null 2>&1
}

wait_for_ros_master() {
  timeout_s="${1:-40}"
  start_wait=$(date +%s)
  while ! ros_master_ready
  do
    if ((($(date +%s) - start_wait) >= timeout_s))
    then
      return 1
    fi
    sleep 1
  done
  return 0
}

topic_ready() {
  rostopic info "$1" >/dev/null 2>&1
}

wait_for_topic() {
  topic_name="$1"
  timeout_s="${2:-40}"
  start_wait=$(date +%s)
  while ! topic_ready "$topic_name"
  do
    if ((($(date +%s) - start_wait) >= timeout_s))
    then
      echo "[LAUNCH SCRIPT] ERROR: Timed out waiting for topic $topic_name"
      return 1
    fi
    sleep 1
  done
  return 0
}

wait_for_planner_ready() {
  [ -z "$planner_ready_topic" ] && return 0
  timeout_s="${1:-45}"
  start_wait=$(date +%s)
  while ((($(date +%s) - start_wait) < timeout_s))
  do
    if ! ps -p "$COMP_PID" >/dev/null 2>&1
    then
      return 1
    fi
    if timeout 2 rostopic echo -n 1 "$planner_ready_topic" 2>/dev/null | grep -Eiq 'data:[[:space:]]*(true|1)'
    then
      benchmark_stage planner_ready
      return 0
    fi
    sleep 1
  done
  echo "[LAUNCH SCRIPT] ERROR: Timed out waiting for planner ready topic $planner_ready_topic"
  return 1
}

wait_for_sim_topics() {
  wait_for_topic /kingfisher/dodgeros_pilot/state 45 || return 1
  wait_for_topic /kingfisher/dodgeros_pilot/groundtruth/obstacles 45 || return 1
  wait_for_topic /kingfisher/dodgeros_pilot/groundtruth/dynamic_obstacles 45 || return 1
  wait_for_topic /kingfisher/dodgeros_pilot/unity/depth 45 || return 1
  return 0
}

publish_empty_control() {
  topic_name="$1"
  minimum_subscribers="${2:-1}"
  connect_timeout="${3:-5}"
  acknowledgement="${4:-}"
  if [ "$benchmark_mode" = "1" ]
  then
    control_args=(
      --topic "$topic_name"
      --type empty
      --min-subscribers "$minimum_subscribers"
      --connect-timeout "$connect_timeout"
    )
    if [ -n "$acknowledgement" ]
    then
      control_args+=(--ack "$acknowledgement" --ack-timeout "$connect_timeout")
    fi
    "$python_bin" ./envtest/ros/publish_control_message.py "${control_args[@]}"
  else
    rostopic pub "$topic_name" std_msgs/Empty "{}" --once
  fi
}

publish_bool_control() {
  topic_name="$1"
  value="$2"
  minimum_subscribers="${3:-1}"
  connect_timeout="${4:-5}"
  acknowledgement="${5:-}"
  if [ "$benchmark_mode" = "1" ]
  then
    control_args=(
      --topic "$topic_name"
      --type bool
      --value "$value"
      --min-subscribers "$minimum_subscribers"
      --connect-timeout "$connect_timeout"
    )
    if [ -n "$acknowledgement" ]
    then
      control_args+=(--ack "$acknowledgement" --ack-timeout "$connect_timeout")
    fi
    "$python_bin" ./envtest/ros/publish_control_message.py "${control_args[@]}"
  else
    rostopic pub "$topic_name" std_msgs/Bool "data: $value" --once
  fi
}

wait_for_process_exit() {
  process_name="$1"
  timeout_s="${2:-10}"
  start_wait=$(date +%s)
  while pgrep -x "$process_name" >/dev/null
  do
    if ((($(date +%s) - start_wait) >= timeout_s))
    then
      return 1
    fi
    sleep 1
  done
  return 0
}

wait_for_pid_exit() {
  process_pid="$1"
  timeout_s="${2:-15}"
  start_wait=$(date +%s)
  while kill -0 "$process_pid" 2>/dev/null
  do
    if ((($(date +%s) - start_wait) >= timeout_s))
    then
      return 1
    fi
    sleep 1
  done
  return 0
}

simulator_stack_running() {
  local process_name
  for process_name in roslaunch visionsim_node flight_render vitfly-unity.x86_64 RPG_Flightmare.x86_64 dodgeros_pilot roscore rosmaster rosout gzserver gzclient
  do
    if pgrep -x "$process_name" >/dev/null
    then
      return 0
    fi
  done
  return 1
}

wait_for_simulator_shutdown() {
  timeout_s="${1:-15}"
  start_wait=$(date +%s)
  while simulator_stack_running || ros_master_ready
  do
    if ((($(date +%s) - start_wait) >= timeout_s))
    then
      return 1
    fi
    sleep 1
  done
  return 0
}

signal_simulator_stack() {
  signal_name="$1"
  killall "-$signal_name" roslaunch visionsim_node flight_render vitfly-unity.x86_64 \
    RPG_Flightmare.x86_64 dodgeros_pilot roscore rosmaster rosout gzserver gzclient \
    RPG_Flightmare. rviz 2>/dev/null
  return 0
}

ensure_environment_exists() {
  environment_dir="$FLIGHTMARE_PATH/flightpy/configs/vision/$VITFLY_ENV_LEVEL/$VITFLY_ENV_FOLDER"
  if [ ! -f "$environment_dir/static_obstacles.csv" ] || \
     [ ! -f "$environment_dir/dynamic_obstacles.yaml" ] || \
     [ ! -f "$environment_dir/astar_path.csv" ]
  then
    echo "[LAUNCH SCRIPT] ERROR: Missing dynamic A* environment assets in $environment_dir"
    return 1
  fi
  return 0
}

launch_simulator() {
  if [ "$reuse_simulator" = "1" ]
  then
    wait_for_ros_master 45 || return 1
    wait_for_sim_topics || return 1
    benchmark_stage simulator_ready
    return 0
  fi
  if [ "$benchmark_mode" != "1" ] && pgrep -x visionsim_node >/dev/null && ros_master_ready
  then
    ROS_PID=""
    wait_for_sim_topics || return 1
    return 0
  fi

  if pgrep -x visionsim_node >/dev/null
  then
    echo "[LAUNCH SCRIPT] Found stale visionsim_node without ROS master, cleaning it before launch."
    killall -9 visionsim_node flight_render vitfly-unity.x86_64 RPG_Flightmare.x86_64 dodgeros_pilot 2>/dev/null
    wait_for_process_exit visionsim_node 10
  fi

  roslaunch envsim visionenv_sim.launch render:=True gui:=False \
    rviz:=$rviz_enabled publish_rgb:=$publish_rgb \
    publish_optical_flow:=$publish_optical_flow $realtimefactor &
  ROS_PID="$!"
  echo $ROS_PID

  if ! wait_for_ros_master 45
  then
    echo "[LAUNCH SCRIPT] ERROR: ROS master did not become ready after launching simulator."
    return 1
  fi

  sleep 5
  wait_for_sim_topics || return 1
  benchmark_stage simulator_ready
}

benchmark_stage() {
  if [ "$benchmark_mode" = "1" ]
  then
    echo "[BENCHMARK_STAGE] $1"
  fi
}

stop_simulator() {
  if [ -n "$ROS_PID" ] && kill -0 "$ROS_PID" 2>/dev/null
  then
    kill -SIGINT "$ROS_PID"
    if ! wait_for_pid_exit "$ROS_PID" 15
    then
      kill -SIGTERM "$ROS_PID" 2>/dev/null
      if ! wait_for_pid_exit "$ROS_PID" 5
      then
        kill -SIGKILL "$ROS_PID" 2>/dev/null
        wait_for_pid_exit "$ROS_PID" 5
      fi
    fi
  fi
  ROS_PID=""
}

force_stop_simulator() {
  stop_simulator
  if simulator_stack_running || ros_master_ready
  then
    echo "[LAUNCH SCRIPT] Cleaning residual simulator processes."
    signal_simulator_stack INT
    wait_for_simulator_shutdown 15
  fi
  if simulator_stack_running || ros_master_ready
  then
    signal_simulator_stack TERM
    wait_for_simulator_shutdown 5
  fi
  if simulator_stack_running || ros_master_ready
  then
    signal_simulator_stack KILL
    wait_for_simulator_shutdown 5
  fi
  if simulator_stack_running || ros_master_ready
  then
    echo "[LAUNCH SCRIPT] ERROR: Simulator processes did not shut down completely."
    return 1
  fi
  return 0
}

prepare_pilot_for_rollout() {
  local attempt
  local maximum_attempts=2
  if [ "$benchmark_mode" = "1" ] && [ "$reuse_simulator" != "1" ]
  then
    maximum_attempts=1
  fi
  for attempt in $(seq 1 "$maximum_attempts")
  do
    echo "[LAUNCH SCRIPT] Preparing low-level pilot (attempt $attempt/$maximum_attempts)"
    benchmark_stage pilot_prepare_start
    if ! publish_empty_control /kingfisher/dodgeros_pilot/off 1 5 pilot_off && \
       [ "$benchmark_mode" = "1" ]
    then
      return 1
    fi
    if [ "$benchmark_mode" != "1" ]; then sleep 1; fi
    if [ "$reuse_simulator" = "1" ]
    then
      reset_response=""
      if ! rosparam set /kingfisher/dodgeros_pilot/dynamic_phase_seed "$VITFLY_DYNAMIC_PHASE_SEED" || \
         ! reset_response=$(rosservice call /kingfisher/dodgeros_pilot/reset_benchmark "{}") || \
         ! printf '%s\n' "$reset_response" | grep -Eiq 'success:[[:space:]]*(true|1)'
      then
        echo "[LAUNCH SCRIPT] ERROR: benchmark simulator reset service failed."
        return 1
      fi
    else
      if ! publish_empty_control /kingfisher/dodgeros_pilot/reset_sim 1 5 pilot_reset && \
         [ "$benchmark_mode" = "1" ]
      then
        return 1
      fi
    fi
    if [ "$benchmark_mode" != "1" ]; then sleep 1; fi
    if ! publish_bool_control /kingfisher/dodgeros_pilot/enable true 1 5 pilot_enabled && \
       [ "$benchmark_mode" = "1" ]
    then
      return 1
    fi
    if [ "$benchmark_mode" != "1" ]; then sleep 1; fi
    if ! publish_empty_control /kingfisher/dodgeros_pilot/start 1 5 && \
       [ "$benchmark_mode" = "1" ]
    then
      return 1
    fi

    if "$python_bin" ./envtest/ros/wait_for_pilot_hover.py --timeout 30
    then
      benchmark_stage pilot_ready
      return 0
    fi

    echo "[LAUNCH SCRIPT] Pilot did not reach hover on attempt $attempt."
    if [ "$attempt" -lt "$maximum_attempts" ]
    then
      force_stop_simulator || return 1
      launch_simulator || return 1
    fi
  done
  return 1
}

stop_controller() {
  if [ -z "$COMP_PID" ]
  then
    return
  fi
  if kill -0 "$COMP_PID" 2>/dev/null
  then
    publish_empty_control /kingfisher/finish_navigation 1 1 >/dev/null 2>&1 || true
    if ! wait_for_pid_exit "$COMP_PID" 5
    then
      kill -SIGINT "$COMP_PID" 2>/dev/null
      if ! wait_for_pid_exit "$COMP_PID" 5
      then
        kill -SIGTERM "$COMP_PID" 2>/dev/null
        if ! wait_for_pid_exit "$COMP_PID" 3
        then
          kill -SIGKILL "$COMP_PID" 2>/dev/null
          wait_for_pid_exit "$COMP_PID" 3
        fi
      fi
    fi
  fi
  wait "$COMP_PID" 2>/dev/null
  COMP_PID=""
}

stop_evaluator() {
  if [ -z "$PY_PID" ]
  then
    return
  fi
  if kill -0 "$PY_PID" 2>/dev/null
  then
    kill -SIGINT "$PY_PID" 2>/dev/null
    if ! wait_for_pid_exit "$PY_PID" 5
    then
      kill -SIGTERM "$PY_PID" 2>/dev/null
      if ! wait_for_pid_exit "$PY_PID" 3
      then
        kill -SIGKILL "$PY_PID" 2>/dev/null
        wait_for_pid_exit "$PY_PID" 3
      fi
    fi
  fi
  wait "$PY_PID" 2>/dev/null
  PY_PID=""
}

write_rollout_failure_summary() {
  failure_reason="$1"
  cat > ./envtest/ros/summary.yaml <<EOF
Success: false
goal_reached: false
collision: false
collision_count: 0
termination_reason: $failure_reason
EOF
}

summary_file_complete() {
  [ -s ./envtest/ros/summary.yaml ] && \
    grep -q "termination_reason:" ./envtest/ros/summary.yaml
}

wait_for_evaluator_result() {
  timeout_s="${1:-5}"
  start_wait=$(date +%s)
  while (($(date +%s) - start_wait < timeout_s))
  do
    if summary_file_complete
    then
      return 0
    fi
    sleep 1
  done
  return 1
}

simulator_error_exit() {
  if [ "$benchmark_mode" = "1" ]
  then
    exit 3
  fi
  exit 1
}

cleanup_on_exit() {
  original_status="$1"
  if ((cleanup_in_progress))
  then
    return
  fi
  cleanup_in_progress=1
  trap - EXIT INT TERM

  benchmark_stage cleanup_start
  stop_evaluator
  stop_controller
  if [ "$benchmark_mode" = "1" ] && [ "$reuse_simulator" != "1" ]
  then
    if ! force_stop_simulator && [ "$original_status" -eq 0 ]
    then
      original_status=3
    fi
  else
    stop_simulator
  fi
  benchmark_stage cleanup_finished
  exit "$original_status"
}

handle_signal() {
  echo "[LAUNCH SCRIPT] Interrupted; cleaning current rollout."
  exit 130
}

trap handle_signal INT TERM
trap 'cleanup_on_exit $?' EXIT

if ((random_env))
then
  ROS_PID=""
else
  export VITFLY_ENV_LEVEL="${VITFLY_ENV_LEVEL:-$env_level}"
  export VITFLY_ENV_FOLDER="${VITFLY_ENV_FOLDER:-environment_0}"
  export VITFLY_ENV_SEED="${VITFLY_ENV_SEED:-10}"
  export VITFLY_DYNAMIC_PHASE_SEED="${phase_seed_override:-$VITFLY_ENV_SEED}"
  ensure_environment_exists || simulator_error_exit
  if [ "$benchmark_mode" = "1" ] && [ "$reuse_simulator" != "1" ]
  then
    force_stop_simulator || simulator_error_exit
  fi
  launch_simulator || simulator_error_exit
fi

SUMMARY_FILE="${VITFLY_EVALUATION_PATH:-evaluation.yaml}"
mkdir -p "$(dirname "$SUMMARY_FILE")"
echo "" > $SUMMARY_FILE

# generate datetime string to label summary folders with in evaluation_node.py
datetime=$(date '+d%m_%d_t%H_%M')

relaunch_sim=0
batch_failed=0
batch_infrastructure_failed=0

for i in $(eval echo {1..$N})
do
  if ((random_env))
  then
    env_id=$(( (i - 1) % env_count ))
    export VITFLY_ENV_LEVEL="$env_level"
    export VITFLY_ENV_FOLDER="environment_$env_id"
    export VITFLY_ENV_SEED="$((10 + env_id))"
    export VITFLY_DYNAMIC_PHASE_SEED="${phase_seed_override:-$((phase_seed_base + i - 1))}"
    echo "[LAUNCH SCRIPT] Using environment $VITFLY_ENV_LEVEL/$VITFLY_ENV_FOLDER seed=$VITFLY_ENV_SEED phase_seed=$VITFLY_DYNAMIC_PHASE_SEED"
    ensure_environment_exists || simulator_error_exit
    force_stop_simulator || simulator_error_exit
    launch_simulator || simulator_error_exit
  fi

  # Reset the simulator if needed
  if ((relaunch_sim))
  then
      echo
      echo
      echo
      echo
      echo RELAUNCHING SIMULATOR ON RUN $i
      echo
      echo
      echo
      echo

    # reset flag and kill everything to restart
    relaunch_sim=0
    force_stop_simulator || simulator_error_exit

    # Launch the simulator, unless it is already running
    launch_simulator || simulator_error_exit

  fi

  if ! prepare_pilot_for_rollout
  then
    echo "[LAUNCH SCRIPT] ERROR: Pilot never reached hover for rollout $i; no trajectory was created."
    simulator_error_exit
  fi

  export ROLLOUT_NAME="rollout_""$i"
  echo "$ROLLOUT_NAME"

  rm -f ./envtest/ros/summary.yaml ./envtest/ros/.summary.yaml.tmp
  benchmark_stage controller_start
  cd ./envtest/ros/
  "$python_bin" evaluation_node.py ${datetime}_N$i &
  PY_PID="$!"

  if [ -n "$planner_adapter" ]
  then
    echo "[LAUNCH_EVALUATION] Starting planner adapter $planner_adapter"
    "$planner_launch" &
  else
    "$python_bin" run_competition.py $run_competition_args --des_vel "$des_vel" \
      --offset "$offset" --model_path "$model_path" &
  fi
  COMP_PID="$!"
  cd -

  if [ "$benchmark_mode" = "1" ]
  then
    echo
    echo [LAUNCH_EVALUATION] Sending start navigation command
    echo
    if ! wait_for_planner_ready 45
    then
      echo "[LAUNCH_EVALUATION] Planner did not become ready."
      batch_infrastructure_failed=1
      stop_evaluator
      write_rollout_failure_summary simulator_error
    elif ! publish_empty_control /kingfisher/start_navigation 2 30
    then
      if ! ps -p "$COMP_PID" > /dev/null
      then
        echo "[LAUNCH_EVALUATION] Controller exited before navigation could start."
        batch_failed=1
        stop_evaluator
        write_rollout_failure_summary controller_error
      else
        simulator_error_exit
      fi
    else
      benchmark_stage navigation_started
    fi
  else
    wait_for_topic /kingfisher/start_navigation 30 || simulator_error_exit
  fi

  start_time=$(date +%s)
  # Wait until the evaluation script has finished
  while ps -p $PY_PID > /dev/null
  do
    if ! ros_master_ready || \
       ! topic_ready /kingfisher/dodgeros_pilot/state || \
       ! topic_ready /kingfisher/dodgeros_pilot/unity/depth
    then
      echo "[LAUNCH_EVALUATION] Simulator topics disappeared during rollout."
      batch_infrastructure_failed=1
      stop_evaluator
      write_rollout_failure_summary simulator_error
      break
    fi
    if ! ps -p "$COMP_PID" > /dev/null
    then
      controller_returncode=0
      wait "$COMP_PID" 2>/dev/null || controller_returncode=$?
      COMP_PID=""
      echo "[LAUNCH_EVALUATION] Controller exited with code $controller_returncode; waiting for evaluator result."
      if [ "$controller_returncode" -eq 0 ] && wait_for_evaluator_result 5
      then
        echo "[LAUNCH_EVALUATION] Evaluator result is complete; treating controller exit as normal shutdown."
        break
      fi
      if [ "$controller_returncode" -eq 3 ]
      then
        echo "[LAUNCH_EVALUATION] Controller reported an input-pipeline failure."
        batch_infrastructure_failed=1
        stop_evaluator
        write_rollout_failure_summary input_pipeline_error
      else
        echo "[LAUNCH_EVALUATION] Controller exited before evaluator completed."
        batch_failed=1
        stop_evaluator
        write_rollout_failure_summary controller_error
      fi
      break
    fi
    if [ "$benchmark_mode" = "1" ]
    then
      sleep 1
    else
      echo
      echo [LAUNCH_EVALUATION] Sending start navigation command
      echo
      publish_empty_control /kingfisher/start_navigation
      sleep 2
    fi

    # if the current iteration has surpassed the time limit, something went wrong (possibly: [Pipeline]     Bridge failed!). Kill the simulator.
    if ((($(date +%s) - start_time) >= 300))
    then
      echo
      echo
      echo
      echo
      echo "Time limit exceeded. Exiting evaluation script loop."
      echo
      echo
      echo
      echo
      stop_evaluator
      batch_infrastructure_failed=1
      write_rollout_failure_summary simulator_error
      break
    fi

  done

  if [ -n "$PY_PID" ]
  then
    wait "$PY_PID" 2>/dev/null
    PY_PID=""
  fi

  cat "$SUMMARY_FILE" "./envtest/ros/summary.yaml" > "tmp.yaml"
  mv "tmp.yaml" "$SUMMARY_FILE"
  benchmark_stage rollout_finished

  benchmark_stage cleanup_start
  stop_controller
  if [ "$benchmark_mode" = "1" ] && [ "$reuse_simulator" != "1" ]
  then
    force_stop_simulator || batch_infrastructure_failed=1
  elif ((random_env))
  then
    stop_simulator
  fi
  benchmark_stage cleanup_finished

  if ((batch_infrastructure_failed))
  then
    exit 3
  fi
done

if ((batch_failed))
then
  exit 2
fi

if [ "$2" = "state" ]
then
  echo "[LAUNCH SCRIPT] Curating the completed state-collection batch."
  "$python_bin" ./envtest/ros/curate_dataset.py ./envtest/ros/train_set \
    --evaluation "$SUMMARY_FILE" --latest "$N" --apply || exit 1
fi
