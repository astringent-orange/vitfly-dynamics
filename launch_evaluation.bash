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
env_count="${VITFLY_ENV_COUNT:-101}"
env_level="${VITFLY_ENV_LEVEL:-dynamic_astar_medium}"
des_vel="${VITFLY_DES_VEL:-4.0}"
offset="${VITFLY_OFFSET:-0}"
model_path="${VITFLY_MODEL_PATH:-}"
phase_seed_override="${VITFLY_DYNAMIC_PHASE_SEED:-}"
phase_seed_base="${VITFLY_DYNAMIC_PHASE_SEED_BASE:-1000}"

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
  elif [[ "$arg" == model_type=* || "$arg" == frame_offset=* ]]
  then
    echo "[LAUNCH SCRIPT] model_type/frame_offset are obsolete; use offset=0,1,2"
    exit 1
  fi
done

if ! [[ "$offset" =~ ^[012]$ ]]
then
  echo "[LAUNCH SCRIPT] offset must be 0, 1, or 2, got: $offset"
  exit 1
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
  realtimefactor=""
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

wait_for_sim_topics() {
  wait_for_topic /kingfisher/dodgeros_pilot/state 45 || return 1
  wait_for_topic /kingfisher/dodgeros_pilot/groundtruth/obstacles 45 || return 1
  wait_for_topic /kingfisher/dodgeros_pilot/groundtruth/dynamic_obstacles 45 || return 1
  wait_for_topic /kingfisher/dodgeros_pilot/unity/depth 45 || return 1
  return 0
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
  if pgrep -x visionsim_node >/dev/null && ros_master_ready
  then
    ROS_PID=""
    wait_for_sim_topics || return 1
    return 0
  fi

  if pgrep -x visionsim_node >/dev/null
  then
    echo "[LAUNCH SCRIPT] Found stale visionsim_node without ROS master, cleaning it before launch."
    killall -9 visionsim_node flight_render dodgeros_pilot 2>/dev/null
    wait_for_process_exit visionsim_node 10
  fi

  roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=$rviz_enabled $realtimefactor &
  ROS_PID="$!"
  echo $ROS_PID

  if ! wait_for_ros_master 45
  then
    echo "[LAUNCH SCRIPT] ERROR: ROS master did not become ready after launching simulator."
    return 1
  fi

  sleep 10
  wait_for_sim_topics || return 1
}

stop_simulator() {
  if [ $ROS_PID ]
  then
    kill -SIGINT "$ROS_PID"
    sleep 3
    ROS_PID=""
  fi
}

force_stop_simulator() {
  stop_simulator
  killall -9 roslaunch visionsim_node flight_render dodgeros_pilot roscore rosmaster rosout gzserver gzclient RPG_Flightmare. rviz 2>/dev/null
  wait_for_process_exit visionsim_node 10
  sleep 10
}

prepare_pilot_for_rollout() {
  local attempt
  for attempt in 1 2
  do
    echo "[LAUNCH SCRIPT] Preparing low-level pilot (attempt $attempt/2)"
    rostopic pub /kingfisher/dodgeros_pilot/off std_msgs/Empty "{}" --once
    sleep 1
    rostopic pub /kingfisher/dodgeros_pilot/reset_sim std_msgs/Empty "{}" --once
    sleep 1
    rostopic pub /kingfisher/dodgeros_pilot/enable std_msgs/Bool "data: true" --once
    sleep 1
    rostopic pub /kingfisher/dodgeros_pilot/start std_msgs/Empty "{}" --once

    if python3 ./envtest/ros/wait_for_pilot_hover.py --timeout 30
    then
      return 0
    fi

    echo "[LAUNCH SCRIPT] Pilot did not reach hover on attempt $attempt."
    if [ "$attempt" -eq 1 ]
    then
      force_stop_simulator
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
  if ps -p "$COMP_PID" > /dev/null
  then
    rostopic pub /kingfisher/finish_navigation std_msgs/Empty "{}" --once >/dev/null 2>&1
    for _ in $(seq 1 5)
    do
      if ! ps -p "$COMP_PID" > /dev/null
      then
        return
      fi
      sleep 1
    done
    kill -SIGINT "$COMP_PID"
  fi
}

if ((random_env))
then
  ROS_PID=""
else
  export VITFLY_ENV_LEVEL="${VITFLY_ENV_LEVEL:-$env_level}"
  export VITFLY_ENV_FOLDER="${VITFLY_ENV_FOLDER:-environment_0}"
  export VITFLY_ENV_SEED="${VITFLY_ENV_SEED:-10}"
  export VITFLY_DYNAMIC_PHASE_SEED="${phase_seed_override:-$VITFLY_ENV_SEED}"
  ensure_environment_exists || exit 1
  launch_simulator || exit 1
fi

SUMMARY_FILE="evaluation.yaml"
echo "" > $SUMMARY_FILE

# generate datetime string to label summary folders with in evaluation_node.py
datetime=$(date '+d%m_%d_t%H_%M')

relaunch_sim=0

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
    ensure_environment_exists || exit 1
    force_stop_simulator
    launch_simulator || exit 1
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
    force_stop_simulator

    # Launch the simulator, unless it is already running
    launch_simulator || exit 1

  fi

  if ! prepare_pilot_for_rollout
  then
    echo "[LAUNCH SCRIPT] ERROR: Pilot never reached hover for rollout $i; no trajectory was created."
    exit 1
  fi

  export ROLLOUT_NAME="rollout_""$i"
  echo "$ROLLOUT_NAME"

  cd ./envtest/ros/
  python3 evaluation_node.py ${datetime}_N$i &
  PY_PID="$!"

  python3 run_competition.py $run_competition_args --des_vel "$des_vel" \
    --offset "$offset" --model_path "$model_path" &
  COMP_PID="$!"
  cd -

  wait_for_topic /kingfisher/start_navigation 30 || exit 1

  start_time=$(date +%s)

  # Wait until the evaluation script has finished
  while ps -p $PY_PID > /dev/null
  do
    echo
    echo [LAUNCH_EVALUATION] Sending start navigation command
    echo
    rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" --once
    sleep 2

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
      kill -SIGINT $PY_PID
      relaunch_sim=1
      break
    fi

  done

  cat "$SUMMARY_FILE" "./envtest/ros/summary.yaml" > "tmp.yaml"
  mv "tmp.yaml" "$SUMMARY_FILE"

  stop_controller
  if ((random_env))
  then
    stop_simulator
  fi
done

if [ $ROS_PID ]
then
  kill -SIGINT "$ROS_PID"
fi

if [ "$2" = "state" ]
then
  echo "[LAUNCH SCRIPT] Curating the completed state-collection batch."
  python3 ./envtest/ros/curate_dataset.py ./envtest/ros/train_set \
    --evaluation "$SUMMARY_FILE" --latest "$N" --apply || exit 1
fi
