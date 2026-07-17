#!/bin/bash

if [[ ! -f "$(pwd)/setup_ros.bash" ]]
then
  echo "please launch from the repository folder!"
  exit 1
fi

project_path="$PWD"
echo "$project_path"

echo "Using apt to install dependencies..."
echo "Will ask for sudo permissions:"
sudo apt update
sudo apt install -y --no-install-recommends build-essential cmake libzmqpp-dev libopencv-dev unzip python3-catkin-tools

echo "Ignoring unused Flightmare folders!"
touch flightmare/flightros/CATKIN_IGNORE

echo "Creating envtest/ros/train_set folder to store images and telemetry during flights..."
mkdir -p "$project_path/envtest/ros/train_set"

echo "Done!"
echo "Have a safe flight!"
