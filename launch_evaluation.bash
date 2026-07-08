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
env_count="${VITFLY_ENV_COUNT:-10}"
env_level="${VITFLY_ENV_LEVEL:-dynamic_astar_medium}"

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
  fi
done

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

launch_simulator() {
  if [ -z $(pgrep visionsim_node) ]
  then
    roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=$rviz_enabled $realtimefactor &
    ROS_PID="$!"
    echo $ROS_PID
    sleep 10
  else
    ROS_PID=""
  fi
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
  killall -9 roscore rosmaster rosout gzserver gzclient RPG_Flightmare. 2>/dev/null
  sleep 10
}

if ((random_env))
then
  force_stop_simulator
else
  export VITFLY_ENV_LEVEL="${VITFLY_ENV_LEVEL:-$env_level}"
  export VITFLY_ENV_FOLDER="${VITFLY_ENV_FOLDER:-environment_0}"
  export VITFLY_ENV_SEED="${VITFLY_ENV_SEED:-10}"
  launch_simulator
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
    echo "[LAUNCH SCRIPT] Using environment $VITFLY_ENV_LEVEL/$VITFLY_ENV_FOLDER seed=$VITFLY_ENV_SEED"
    launch_simulator
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
    launch_simulator

  fi

  start_time=$(date +%s)

  # Publish simulator reset
  rostopic pub /kingfisher/dodgeros_pilot/off std_msgs/Empty "{}" --once
  rostopic pub /kingfisher/dodgeros_pilot/reset_sim std_msgs/Empty "{}" --once
  rostopic pub /kingfisher/dodgeros_pilot/enable std_msgs/Bool "data: true" --once
  rostopic pub /kingfisher/dodgeros_pilot/start std_msgs/Empty "{}" --once

  export ROLLOUT_NAME="rollout_""$i"
  echo "$ROLLOUT_NAME"

  cd ./envtest/ros/
  python3 evaluation_node.py ${datetime}_N$i &
  PY_PID="$!"

  python3 run_competition.py $run_competition_args --des_vel 5.0 --model_type "ViTLSTM" --model_path ../../models/ViTLSTM_model.pth &
  COMP_PID="$!"
  # python3 run_competition.py $run_competition_args --des_vel 5.0 --model_type "ViTLSTM" --model_path ../../models/model_000396.pth &
  # COMP_PID="$!"

  cd -

  sleep 2

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

  kill -SIGINT "$COMP_PID"
  if ((random_env))
  then
    stop_simulator
    force_stop_simulator
  fi
done

if [ $ROS_PID ]
then
  kill -SIGINT "$ROS_PID"
fi
