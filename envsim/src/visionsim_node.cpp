#include <ros/ros.h>

#include "dodgeros_msgs/QuadState.h"
#include "envsim/visionsim.hpp"
#include "envsim_msgs/ObstacleArray.h"

#include <cstdlib>
#include <sstream>
#include "nav_msgs/Odometry.h"
#include "rosgraph_msgs/Clock.h"

using namespace agi;

VisionSim::VisionSim(const ros::NodeHandle &nh, const ros::NodeHandle &pnh)
  : nh_(nh), pnh_(pnh), frame_id_(0) {
  // Logic subscribers
  reset_sub_ = pnh_.subscribe("reset_sim", 1, &VisionSim::resetCallback, this);
  reset_benchmark_service_ = pnh_.advertiseService(
    "reset_benchmark", &VisionSim::resetBenchmarkCallback, this);
  reset_dynamic_phases_service_ = pnh_.advertiseService(
    "reset_dynamic_phases", &VisionSim::resetDynamicPhasesCallback, this);

  // Publishers
  clock_pub_ = nh_.advertise<rosgraph_msgs::Clock>("/clock", 1);
  odometry_pub_ = pnh_.advertise<nav_msgs::Odometry>("groundtruth/odometry", 1);
  state_pub_ = pnh_.advertise<dodgeros_msgs::QuadState>("groundtruth/state", 1);

  image_transport::ImageTransport it(pnh_);

  pnh_.param("publish_rgb", publish_rgb_, true);
  pnh_.param("publish_optical_flow", publish_optical_flow_, true);

  obstacle_pub_ =
    pnh_.advertise<envsim_msgs::ObstacleArray>("groundtruth/obstacles", 1);
  dynamic_obstacle_pub_ =
    pnh_.advertise<envsim_msgs::ObstacleArray>("groundtruth/dynamic_obstacles", 1);

  if (publish_rgb_) image_pub_ = it.advertise("unity/image", 1);
  depth_pub_ = it.advertise("unity/depth", 1);
  if (publish_optical_flow_)
    opticalflow_pub_ = it.advertise("unity/opticalflow", 1);

  ROS_INFO("Vision image outputs: depth=on rgb=%s optical_flow=%s",
           publish_rgb_ ? "on" : "off",
           publish_optical_flow_ ? "on" : "off");

  ros_pilot_.getQuadrotor(&quad_);
  simulator_.updateQuad(quad_);
  simulator_.addModel(ModelInit{quad_});
  simulator_.addModel(ModelMotor{quad_});

  std::string low_level_ctrl;
  pnh_.getParam("render", render_);
  pnh_.getParam("agi_param_dir", agi_param_directory_);
  pnh_.getParam("ros_param_dir", ros_param_directory_);
  pnh_.getParam("real_time_factor", real_time_factor_);
  const char* env_level = std::getenv("VITFLY_ENV_LEVEL");
  const char* env_folder = std::getenv("VITFLY_ENV_FOLDER");
  pnh_.setParam("active_env_level", env_level ? std::string(env_level) : "");
  pnh_.setParam("active_env_folder", env_folder ? std::string(env_folder) : "");
  pnh_.setParam("real_time_factor", real_time_factor_);
  const bool got_directory =
    pnh_.getParam("agi_param_dir", agi_param_directory_);

  simulator_.addModel(ModelThrustTorqueSimple{quad_});
  simulator_.addModel(ModelRigidBody{quad_});

  std::string env_cfg_file =
    getenv("FLIGHTMARE_PATH") +
    std::string("/flightpy/configs/vision/config.yaml");

  vision_env_ptr_ = std::make_unique<flightlib::VisionEnv>(env_cfg_file, 0);
  if (render_) {
    std::string camera_config = ros_param_directory_ + "/camera_config.yaml";
    // if (!(std::filesystem::exists(camera_config))) {
    //   ROS_ERROR("Configuration file [%s] does not exists.",
    //             camera_config.c_str());
    // }
    YAML::Node cfg_node = YAML::LoadFile(camera_config);
    vision_env_ptr_->configCamera(cfg_node);
    vision_env_ptr_->setUnity(render_);
    vision_env_ptr_->connectUnity();
  }

  // wait until Unity is up
  ros::WallDuration(1.0).sleep();
  t_start_ = ros::WallTime::now();
  sim_thread_ = std::thread(&VisionSim::simLoop, this);
}

VisionSim::~VisionSim() {
  stop_requested_.store(true);
  if (sim_thread_.joinable()) sim_thread_.join();
  if (render_thread_.joinable()) render_thread_.join();
  if (render_ && vision_env_ptr_) vision_env_ptr_->disconnectUnity();
}

bool VisionSim::resetSimulation(uint32_t phase_seed, std::vector<Scalar>* phases) {
  ROS_INFO("Resetting simulator!");
  QuadState reset_state;
  {
    const std::lock_guard<std::mutex> lock(sim_mutex_);
    // Keep the externally visible simulated clock monotonic when a reused
    // session resets its local simulator time to zero.
    sim_time_offset_ = std::max(
      sim_time_offset_, last_published_time_ - t_start_.toSec() + sim_dt_);
    simulator_.reset(false);
    simulator_.setCommand(Command(0.0, 0.0, Vector<3>::Zero()));
    simulator_.getState(&reset_state);
  }
  {
    const std::lock_guard<std::mutex> lock(dynamic_objects_mutex_);
    *phases = vision_env_ptr_->resetDynamicObstaclePhases(phase_seed);
    // Change the position of dynamic obstacles if the trigger is set while
    // holding the same lock used by the simulation loop.
    if (vision_env_ptr_->_move_obst_trigger) vision_env_ptr_->move();
  }
  std::ostringstream phase_log;
  phase_log << "Reset dynamic obstacle phases with seed=" << phase_seed << ":";
  for (const Scalar phase : *phases) phase_log << " " << phase;
  ROS_INFO_STREAM(phase_log.str());
  reset_state.t += t_start_.toSec() + sim_time_offset_;
  ROS_INFO_STREAM("Simulator reset complete at external time " << reset_state.t);
  return true;
}

void VisionSim::resetCallback(const std_msgs::EmptyConstPtr &msg) {
  const char* phase_seed_env = std::getenv("VITFLY_DYNAMIC_PHASE_SEED");
  if (phase_seed_env == nullptr || std::string(phase_seed_env).empty()) {
    phase_seed_env = std::getenv("VITFLY_ENV_SEED");
  }
  const uint32_t phase_seed = phase_seed_env == nullptr
    ? 0u : static_cast<uint32_t>(std::strtoul(phase_seed_env, nullptr, 10));
  std::vector<Scalar> phases;
  resetSimulation(phase_seed, &phases);
}

bool VisionSim::resetBenchmarkCallback(std_srvs::Trigger::Request& request,
                                        std_srvs::Trigger::Response& response) {
  (void)request;
  int seed = 0;
  pnh_.param("dynamic_phase_seed", seed, 0);
  std::vector<Scalar> phases;
  response.success = resetSimulation(static_cast<uint32_t>(std::max(seed, 0)), &phases);
  std::ostringstream message;
  message << "phase_seed=" << std::max(seed, 0) << " phases=" << phases.size();
  response.message = message.str();
  return true;
}

bool VisionSim::resetDynamicPhasesCallback(std_srvs::Trigger::Request& request,
                                            std_srvs::Trigger::Response& response) {
  (void)request;
  int seed = 0;
  pnh_.param("dynamic_phase_seed", seed, 0);
  std::vector<Scalar> phases;
  {
    const std::lock_guard<std::mutex> lock(dynamic_objects_mutex_);
    phases = vision_env_ptr_->resetDynamicObstaclePhases(
      static_cast<uint32_t>(std::max(seed, 0)));
  }
  response.success = true;
  std::ostringstream message;
  message << "phase_seed=" << std::max(seed, 0)
          << " phases=" << phases.size();
  response.message = message.str();
  ROS_INFO_STREAM("Reset dynamic obstacle phases before navigation with seed="
                  << std::max(seed, 0));
  return true;
}

void VisionSim::simLoop() {
  while (ros::ok() && !stop_requested_.load()) {
    ros::WallTime t_start_sim = ros::WallTime::now();
    QuadState quad_state;
    Scalar sim_time = 0.0;
    Scalar external_time = 0.0;
    {
      const std::lock_guard<std::mutex> lock(sim_mutex_);
      simulator_.getState(&quad_state);
      sim_time = quad_state.t;
      external_time = quad_state.t + t_start_.toSec() + sim_time_offset_;
      quad_state.t = external_time;
      last_published_time_ = external_time;
    }

    // we add an offset to have realistic timestamps
    rosgraph_msgs::Clock curr_time;
    curr_time.clock.fromSec(quad_state.t);
    clock_pub_.publish(curr_time);

    // sleep for 1ms
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    publishState(quad_state);

    ros_pilot_.getPilot().odometryCallback(quad_state);

    Command cmd = ros_pilot_.getCommand();
    Scalar command_time_origin = 0.0;
    {
      const std::lock_guard<std::mutex> lock(sim_mutex_);
      command_time_origin = t_start_.toSec() + sim_time_offset_;
    }
    cmd.t -= command_time_origin;
    if (cmd.valid()) {
      {
        const std::lock_guard<std::mutex> lock(sim_mutex_);
        simulator_.setCommand(cmd);
      }
    } else {
      Command zero_cmd;
      zero_cmd.t = sim_time;  // quad_state.t;
      zero_cmd.thrusts.setZero();
      {
        const std::lock_guard<std::mutex> lock(sim_mutex_);
        simulator_.setCommand(zero_cmd);
      }
    }
    {
      const std::lock_guard<std::mutex> lock(sim_mutex_);
      if (!simulator_.run(sim_dt_))
        ROS_WARN_THROTTLE(1.0, "Simulation failed!");
    }
    // Render here stuff
    if (render_) {
      if ((step_counter_ + 1) % render_every_n_steps_ == 0) {
        publishImages(quad_state);
        step_counter_ = 0;
      } else {
        step_counter_ += 1;
      }
    }

    {
      const std::lock_guard<std::mutex> lock(dynamic_objects_mutex_);
      // simulate dynamic obstacles
      std::vector<std::shared_ptr<flightlib::UnityObject>> dynamic_objects =
        vision_env_ptr_->getDynamicObjects();
      if (vision_env_ptr_->_dynamic_obstacles_motion ||
          !vision_env_ptr_->_move_obst_trigger) {
        for (int i = 0; i < int(dynamic_objects.size()); i++) {
          dynamic_objects[i]->run(sim_dt_);
        }
      }
      publishObstacles(quad_state);
      publishDynamicObstacles(quad_state);
    }

    Scalar sleep_time = 1.0 / real_time_factor_ * sim_dt_ -
                        (ros::WallTime::now() - t_start_sim).toSec();
    ros::WallDuration(std::max(sleep_time, 0.0)).sleep();
  }
}

void VisionSim::publishState(const QuadState &state) {
  dodgeros_msgs::QuadState msg_state;
  msg_state.header.frame_id = "world";
  msg_state.header.stamp = ros::Time(state.t);
  msg_state.t = state.t;
  msg_state.pose.position = toRosPoint(state.p);
  msg_state.pose.orientation = toRosQuaternion(state.q());
  msg_state.velocity.linear = toRosVector(state.v);
  msg_state.velocity.angular = toRosVector(state.w);
  msg_state.acceleration.linear = toRosVector(state.a);
  msg_state.acceleration.angular = toRosVector(state.tau);

  nav_msgs::Odometry msg_odo;
  msg_odo.header.frame_id = "world";
  msg_odo.header.stamp = ros::Time(state.t);
  msg_odo.pose.pose = msg_state.pose;
  msg_odo.twist.twist = msg_state.velocity;

  odometry_pub_.publish(msg_odo);
  state_pub_.publish(msg_state);
}

void VisionSim::publishObstacles(const QuadState &state) {
  flightlib::QuadState unity_quad_state;
  unity_quad_state.setZero();
  unity_quad_state.p = state.p.cast<flightlib::Scalar>();
  unity_quad_state.qx = state.qx.cast<flightlib::Scalar>();

  vision_env_ptr_->getQuadrotor()->setState(unity_quad_state);

  envsim_msgs::ObstacleArray obstacle_msg;
  obstacle_msg.header.stamp = ros::Time(state.t);
  obstacle_msg.t = state.t;
  obstacle_msg.num = vision_env_ptr_->getNumDetectedObstacles();

  flightlib::Vector<> obstacle_state;
  const int obstacle_obs_dim = 4;
  obstacle_state.resize(obstacle_obs_dim * obstacle_msg.num);
  vision_env_ptr_->getObstacleState(obstacle_state);

  for (int i = 0; i < obstacle_msg.num; i++) {
    envsim_msgs::Obstacle single_obstacle;
    single_obstacle.position.x = obstacle_state[obstacle_obs_dim * i];
    single_obstacle.position.y = obstacle_state[obstacle_obs_dim * i + 1];
    single_obstacle.position.z = obstacle_state[obstacle_obs_dim * i + 2];
    single_obstacle.scale = obstacle_state[obstacle_obs_dim * i + 3];

    obstacle_msg.obstacles.push_back(single_obstacle);
  }
  obstacle_pub_.publish(obstacle_msg);
}


void VisionSim::publishDynamicObstacles(const QuadState &state) {
  std::vector<std::shared_ptr<flightlib::UnityObject>> dynamic_objects =
    vision_env_ptr_->getDynamicObjects();

  envsim_msgs::ObstacleArray obstacle_msg;
  obstacle_msg.header.stamp = ros::Time(state.t);
  obstacle_msg.t = state.t;
  obstacle_msg.num = dynamic_objects.size();

  for (int i = 0; i < int(dynamic_objects.size()); i++) {
    flightlib::Vector<3> quad_pos = state.p.cast<flightlib::Scalar>();
    flightlib::Vector<3> relative_pos = dynamic_objects[i]->getPos() - quad_pos;

    envsim_msgs::Obstacle single_obstacle;
    single_obstacle.position.x = relative_pos.x();
    single_obstacle.position.y = relative_pos.y();
    single_obstacle.position.z = relative_pos.z();
    single_obstacle.scale = dynamic_objects[i]->getScale()[0] / 2.0;

    obstacle_msg.obstacles.push_back(single_obstacle);
  }
  dynamic_obstacle_pub_.publish(obstacle_msg);
}


void VisionSim::publishImages(const QuadState &state) {
  frame_id_ += 1;
  // render the frame
  flightlib::QuadState unity_quad_state;
  unity_quad_state.setZero();
  unity_quad_state.p = state.p.cast<flightlib::Scalar>();
  unity_quad_state.qx = state.qx.cast<flightlib::Scalar>();

  std::shared_ptr<flightlib::Quadrotor> unity_quad =
    vision_env_ptr_->getQuadrotor();
  unity_quad->setState(unity_quad_state);


  const flightlib::FrameID received_frame_id =
    vision_env_ptr_->updateUnity(frame_id_);
  if (received_frame_id != frame_id_) {
    ROS_WARN_THROTTLE(1.0,
                      "Unity frame %llu timed out; skipping stale image data",
                      static_cast<unsigned long long>(frame_id_));
    return;
  }

  // Unity always returns a base RGB layer, but benchmark mode avoids copying,
  // converting and publishing layers that the controller does not consume.
  std::shared_ptr<flightlib::RGBCamera> camera = unity_quad->getCameras()[0];
  if (publish_rgb_) {
    cv::Mat rgb;
    if (camera->getRGBImage(rgb)) {
      sensor_msgs::ImagePtr rgb_msg =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", rgb).toImageMsg();
      rgb_msg->header.stamp = ros::Time(state.t);
      image_pub_.publish(rgb_msg);
    }
  }

  cv::Mat depth;
  if (!camera->getDepthMap(depth)) {
    ROS_WARN_THROTTLE(1.0, "Unity did not return a depth image for this frame");
    return;
  }
  sensor_msgs::ImagePtr depth_msg =
    cv_bridge::CvImage(std_msgs::Header(), "32FC1", depth).toImageMsg();
  depth_msg->header.stamp = ros::Time(state.t);
  depth_pub_.publish(depth_msg);

  if (publish_optical_flow_) {
    cv::Mat optical_flow;
    if (camera->getOpticalFlow(optical_flow)) {
      sensor_msgs::ImagePtr optical_flow_msg =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", optical_flow).toImageMsg();
      optical_flow_msg->header.stamp = ros::Time(state.t);
      opticalflow_pub_.publish(optical_flow_msg);
    }
  }
}


int main(int argc, char **argv) {
  ros::init(argc, argv, "visionsim_node");

  VisionSim vision_sim;

  ros::spin();
  return 0;
}
