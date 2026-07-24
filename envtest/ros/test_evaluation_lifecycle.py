import importlib.util
import os
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def load_evaluation_node():
    rospy = types.ModuleType("rospy")
    rospy.signal_shutdown = lambda _reason: None
    rospy.init_node = lambda *_args, **_kwargs: None

    dodgeros_msgs = types.ModuleType("dodgeros_msgs")
    dodgeros_msgs_msg = types.ModuleType("dodgeros_msgs.msg")
    dodgeros_msgs_msg.QuadState = type("QuadState", (), {})
    envsim_msgs = types.ModuleType("envsim_msgs")
    envsim_msgs_msg = types.ModuleType("envsim_msgs.msg")
    envsim_msgs_msg.ObstacleArray = type("ObstacleArray", (), {})
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Empty = type("Empty", (), {})
    uniplot = types.ModuleType("uniplot")
    uniplot.plot = lambda *_args, **_kwargs: None

    modules = {
        "rospy": rospy,
        "dodgeros_msgs": dodgeros_msgs,
        "dodgeros_msgs.msg": dodgeros_msgs_msg,
        "envsim_msgs": envsim_msgs,
        "envsim_msgs.msg": envsim_msgs_msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "uniplot": uniplot,
    }
    spec = importlib.util.spec_from_file_location(
        "vitfly_test_evaluation_node",
        ROOT / "envtest" / "ros" / "evaluation_node.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module, rospy


class EvaluationLifecycleTest(unittest.TestCase):
    def test_evaluator_persists_result_before_stopping_controller(self):
        module, rospy = load_evaluation_node()
        events = []
        rospy.signal_shutdown = lambda _reason: events.append("shutdown")

        evaluator = types.SimpleNamespace(
            finish_lock=threading.Lock(),
            finished=False,
            is_active=True,
            printSummary=lambda: events.append("summary"),
            finish_pub=types.SimpleNamespace(publish=lambda: events.append("finish")),
            printPlots=lambda: events.append("plots"),
            plots_enabled=False,
        )
        module.Evaluator.publishFinish(evaluator)
        self.assertEqual(events, ["summary", "finish", "shutdown"])

    def test_collision_abort_persists_result_and_finishes_once(self):
        module, rospy = load_evaluation_node()
        events = []
        rospy.get_time = lambda: 4.0
        rospy.signal_shutdown = lambda _reason: events.append("shutdown")
        evaluator = types.SimpleNamespace(
            finish_lock=threading.Lock(),
            finished=False,
            is_active=True,
            goal_reached=False,
            crash=1,
            termination_reason="collision",
            time_array=[1.0],
            writeSummary=lambda _summary: events.append("summary"),
            finish_pub=types.SimpleNamespace(publish=lambda: events.append("finish")),
        )
        module.Evaluator.abortRun(evaluator)
        module.Evaluator.abortRun(evaluator)
        self.assertEqual(events, ["summary", "finish", "shutdown"])
        self.assertTrue(evaluator.finished)
        self.assertFalse(evaluator.is_active)

    def test_dynamic_obstacle_diagnostics_record_encounter_and_clearance(self):
        module, _rospy = load_evaluation_node()
        obstacle = types.SimpleNamespace(
            position=types.SimpleNamespace(x=1.0, y=0.0, z=0.0),
            scale=0.5,
        )
        message = types.SimpleNamespace(t=1.0, obstacles=[obstacle])
        evaluator = types.SimpleNamespace(
            is_active=True,
            finished=False,
            dynamic_last_timestamp=None,
            dynamic_in_encounter=False,
            dynamic_interaction_time=0.0,
            dynamic_clearances=[],
            dynamic_encounter_count=0,
            dynamic_encounter_margin=2.0,
            collision_margin=0.0,
            dynamic_collision=False,
        )
        module.Evaluator.callbackDynamicObstacles(evaluator, message)
        self.assertEqual(evaluator.dynamic_encounter_count, 1)
        self.assertAlmostEqual(evaluator.dynamic_clearances[0], 0.5)
        self.assertFalse(evaluator.dynamic_collision)

        message.t = 2.0
        module.Evaluator.callbackDynamicObstacles(evaluator, message)
        self.assertAlmostEqual(evaluator.dynamic_interaction_time, 1.0)

    def test_benchmark_forces_terminal_plots_off(self):
        module, _rospy = load_evaluation_node()
        with patch.dict(
            os.environ,
            {"VITFLY_BENCHMARK_MODE": "1", "VITFLY_EVAL_PLOTS": "true"},
            clear=False,
        ):
            self.assertFalse(module.terminal_plots_enabled({"plots": True}))

    def test_non_benchmark_honors_plot_configuration_and_override(self):
        module, _rospy = load_evaluation_node()
        with patch.dict(os.environ, {"VITFLY_BENCHMARK_MODE": "0"}, clear=False):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("VITFLY_EVAL_PLOTS", None)
                self.assertTrue(module.terminal_plots_enabled({"plots": True}))
                self.assertFalse(module.terminal_plots_enabled({"plots": False}))
            with patch.dict(os.environ, {"VITFLY_EVAL_PLOTS": "false"}, clear=False):
                self.assertFalse(module.terminal_plots_enabled({"plots": True}))

    def test_controller_callback_state_and_publishers_precede_subscribers(self):
        source = (ROOT / "envtest" / "ros" / "run_competition.py").read_text()
        depth_subscriber = source.index("self.img_sub = rospy.Subscriber")
        for marker in (
            "self.ctr = 0",
            "self.keyboard_input = ''",
            "self.rgb_img = None",
            "self.debug_img1_pub = rospy.Publisher",
            "self.debug_img2_pub = rospy.Publisher",
        ):
            self.assertLess(source.index(marker), depth_subscriber)

    def test_launcher_waits_for_result_before_controller_error(self):
        source = (ROOT / "launch_evaluation.bash").read_text()
        monitor_loop = source.index("while ps -p $PY_PID")
        controller_branch = source.index('if ! ps -p "$COMP_PID"', monitor_loop)
        wait = source.index("wait_for_evaluator_result 5", controller_branch)
        failure = source.index("write_rollout_failure_summary controller_error", controller_branch)
        self.assertLess(wait, failure)

    def test_benchmark_forces_headless_rviz_after_debug_override(self):
        source = (ROOT / "launch_evaluation.bash").read_text()
        debug_override = source.index("if ((force_rviz))")
        benchmark_override = source.index(
            'if [ "$benchmark_mode" = "1" ]',
            debug_override,
        )
        launch = source.index("launch_simulator()")
        self.assertLess(debug_override, benchmark_override)
        self.assertLess(benchmark_override, launch)
        self.assertIn("rviz_enabled=False", source[benchmark_override:launch])

    def test_benchmark_forces_depth_only_image_topics(self):
        source = (ROOT / "launch_evaluation.bash").read_text()
        defaults = source.index("publish_rgb=True")
        benchmark_override = source.index(
            'if [ "$benchmark_mode" = "1" ]',
            defaults,
        )
        launch = source.index("launch_simulator()")
        override_body = source[benchmark_override:launch]
        self.assertIn("publish_rgb=False", override_body)
        self.assertIn("publish_optical_flow=False", override_body)

        simulator_launch = (ROOT / "envsim" / "launch" / "visionenv_sim.launch").read_text()
        self.assertIn('<arg name="publish_rgb" default="True"/>', simulator_launch)
        self.assertIn('<arg name="publish_optical_flow" default="True"/>', simulator_launch)

        simulator_source = (ROOT / "envsim" / "src" / "visionsim_node.cpp").read_text()
        self.assertIn("if (publish_rgb_)", simulator_source)
        self.assertIn("if (publish_optical_flow_)", simulator_source)
        self.assertIn("if (!camera->getDepthMap(depth))", simulator_source)

    def test_automated_state_collection_disables_terminal_plots_by_default(self):
        source = (ROOT / "launch_evaluation.bash").read_text()
        state_block_start = source.index('if [ "$2" = "state" ] && ((!state_human))')
        state_block_end = source.index("# Batch benchmarks", state_block_start)
        state_block = source[state_block_start:state_block_end]
        self.assertIn("export VITFLY_EVAL_PLOTS=false", state_block)
        self.assertIn('if [ -z "${VITFLY_EVAL_PLOTS:-}" ]', state_block)

    def test_benchmark_controller_skips_rgb_and_debug_image_endpoints(self):
        source = (ROOT / "envtest" / "ros" / "run_competition.py").read_text()
        rgb_initialization = source.index("self.rgb_img_sub = None")
        rgb_callback = source.index("def rgb_callback", rgb_initialization)
        rgb_body = source[rgb_initialization:rgb_callback]
        self.assertIn("if not self.benchmark_mode:", rgb_body)
        self.assertIn("/dodgeros_pilot/unity/image", rgb_body)

        debug_initialization = source.index("self.debug_img1_pub = None")
        subscriber_section = source.index("# Logic subscribers", debug_initialization)
        debug_body = source[debug_initialization:subscriber_section]
        self.assertIn("if not self.benchmark_mode:", debug_body)

        debug_publish = source.index("self.debug_img1_pub.publish")
        publish_guard = source.rfind("if not self.benchmark_mode:", 0, debug_publish)
        self.assertGreater(publish_guard, source.index("def img_callback"))

    def test_planner_adapter_waits_ready_before_navigation(self):
        source = (ROOT / "launch_evaluation.bash").read_text()
        planner_start = source.index('"$planner_launch" &')
        ready_wait = source.index("wait_for_planner_ready 45", planner_start)
        start_publish = source.index("publish_empty_control /kingfisher/start_navigation 2 30", ready_wait)
        self.assertLess(planner_start, ready_wait)
        self.assertLess(ready_wait, start_publish)
        self.assertIn("VITFLY_PLANNER_READY_TOPIC", source)

    def test_normal_navigation_waits_for_both_start_subscribers(self):
        source = (ROOT / "launch_evaluation.bash").read_text()
        normal_start = source.index("wait_for_topic /kingfisher/start_navigation 30")
        normal_end = source.index("start_time=$(date +%s)", normal_start)
        normal_body = source[normal_start:normal_end]
        self.assertIn(
            "publish_empty_control /kingfisher/start_navigation 2 5",
            normal_body,
        )

    def test_random_rollouts_do_not_repeat_clean_force_stop(self):
        source = (ROOT / "launch_evaluation.bash").read_text()
        self.assertIn("simulator_needs_cleanup=1", source)
        random_block_start = source.index('if ((random_env))\n  then\n    env_id=')
        random_block_end = source.index("launch_simulator || simulator_error_exit", random_block_start)
        random_block = source[random_block_start:random_block_end]
        self.assertIn("if ((simulator_needs_cleanup))", random_block)
        self.assertNotIn("force_stop_simulator || simulator_error_exit\n    launch_simulator", random_block)

    def test_simulator_waits_for_messages_instead_of_fixed_sleep(self):
        source = (ROOT / "launch_evaluation.bash").read_text()
        self.assertIn("wait_for_message()", source)
        self.assertIn("wait_for_message /kingfisher/dodgeros_pilot/state 45", source)
        self.assertNotIn("sleep 5\n  wait_for_sim_topics", source)

    def test_launcher_records_rollout_stage_timings(self):
        source = (ROOT / "launch_evaluation.bash").read_text()
        self.assertIn("VITFLY_ROLLOUT_TIMING", source)
        self.assertIn("VITFLY_ROLLOUT_TIMING_LOG", source)
        for stage in (
            "prelaunch_cleanup", "simulator_startup", "pilot_prepare",
            "controller_startup", "navigation", "controller_cleanup",
            "simulator_shutdown", "rollout_total",
        ):
            self.assertIn(f'record_rollout_timing "$i" {stage}', source)

    def test_launcher_uses_short_configurable_roslaunch_shutdown_timeouts(self):
        source = (ROOT / "launch_evaluation.bash").read_text()
        self.assertIn('VITFLY_ROS_SIGINT_TIMEOUT:-3', source)
        self.assertIn('VITFLY_ROS_SIGTERM_TIMEOUT:-1', source)
        self.assertIn('--sigint-timeout="$ros_sigint_timeout"', source)
        self.assertIn('--sigterm-timeout="$ros_sigterm_timeout"', source)
        self.assertIn('wait_for_pid_exit "$ROS_PID" 6', source)
        self.assertIn('wait_for_pid_exit "$ROS_PID" 2', source)
        self.assertIn("wait_for_simulator_shutdown 5", source)
        self.assertIn("wait_for_simulator_shutdown 2", source)

    def test_state_collection_defaults_to_direct_hover_with_fallback(self):
        source = (ROOT / "launch_evaluation.bash").read_text()
        self.assertIn('VITFLY_DIRECT_HOVER_START:-1', source)
        self.assertIn('VITFLY_DIRECT_HOVER_READY_TIMEOUT:-10', source)
        self.assertIn('direct_hover_start:=$direct_hover_ros', source)
        self.assertIn(
            "wait_for_bool_true /kingfisher/dodgeros_pilot/direct_hover_ready",
            source,
        )
        self.assertIn(
            "Direct-hover initialization failed; falling back to traditional takeoff.",
            source,
        )
        self.assertIn('export VITFLY_DIRECT_HOVER_START=0', source)

    def test_direct_hover_can_be_disabled_explicitly(self):
        source = (ROOT / "launch_evaluation.bash").read_text()
        self.assertIn('elif [ "$arg" = "traditional_takeoff" ]', source)
        self.assertIn("direct_hover_start=0", source)

    def test_benchmark_disables_per_frame_inference_timing(self):
        source = (ROOT / "envtest" / "ros" / "run_competition.py").read_text()
        self.assertIn('VITFLY_INFERENCE_TIMING_LOGS', source)
        self.assertIn('timing_default = "false" if self.benchmark_mode else "true"', source)
        timing_log = source.index("compute_command_vision_based took")
        guard = source.rfind("if self.inference_timing_logs", 0, timing_log)
        self.assertGreater(guard, source.index("def img_callback"))

    def test_benchmark_controller_validates_depth_without_rejecting_sparse_zero_pixels(self):
        source = (ROOT / "envtest" / "ros" / "run_competition.py").read_text()
        helper = source.index("def prepare_depth_frame")
        callback = source.index("def img_callback")
        helper_body = source[helper:callback]
        self.assertIn('return normalized, "partial_zero"', helper_body)
        self.assertIn('return None, "all_zero"', helper_body)
        self.assertIn("np.isnan(image).any()", helper_body)
        self.assertIn("np.isneginf(image).any()", helper_body)
        self.assertNotIn("expected_shape", helper_body)

    def test_benchmark_controller_has_wall_clock_no_command_watchdog(self):
        source = (ROOT / "envtest" / "ros" / "run_competition.py").read_text()
        self.assertIn('VITFLY_NO_COMMAND_TIMEOUT_SECONDS', source)
        self.assertIn('self.exit_code = 3', source)
        self.assertIn('self.controller_failure_detail = "no_command"', source)
        self.assertIn('time.monotonic()', source)

    def test_checkpoint_loader_uses_weights_only(self):
        source = (ROOT / "envtest" / "ros" / "run_competition.py").read_text()
        self.assertIn("weights_only=True", source)

    def test_benchmark_launcher_emits_stage_markers_and_acknowledgements(self):
        source = (ROOT / "launch_evaluation.bash").read_text()
        for marker in (
            "simulator_ready", "pilot_prepare_start", "pilot_ready",
            "controller_start", "navigation_started", "rollout_finished",
            "cleanup_start", "cleanup_finished",
        ):
            self.assertIn(f"benchmark_stage {marker}", source)
        for ack in ("pilot_off", "pilot_reset", "pilot_enabled"):
            self.assertIn(ack, source)
        self.assertIn("reset_response", source)

    def test_simulator_waits_for_live_data_before_ready(self):
        source = (ROOT / "launch_evaluation.bash").read_text()
        launch_start = source.index("launch_simulator()")
        launch_end = source.index("stop_simulator()", launch_start)
        launch_body = source[launch_start:launch_end]
        self.assertIn("wait_for_sim_topics || return 1", launch_body)
        self.assertIn("wait_for_message /kingfisher/dodgeros_pilot/state 45", source)
        self.assertIn("wait_for_message /kingfisher/dodgeros_pilot/unity/depth 45", source)
        self.assertNotIn("sleep 5", launch_body)
        self.assertNotIn("sleep 10", launch_body)

    def test_benchmark_control_messages_use_connection_aware_publisher(self):
        source = (ROOT / "launch_evaluation.bash").read_text()
        empty_helper = source.index("publish_empty_control()")
        bool_helper = source.index("publish_bool_control()")
        pilot = source.index("prepare_pilot_for_rollout()")
        self.assertIn("publish_control_message.py", source[empty_helper:bool_helper])
        self.assertIn("publish_control_message.py", source[bool_helper:pilot])
        pilot_end = source.index("stop_controller()", pilot)
        pilot_body = source[pilot:pilot_end]
        for topic in ("off", "reset_sim", "enable", "start"):
            self.assertIn(f"dodgeros_pilot/{topic}", pilot_body)

    def test_benchmark_navigation_start_is_published_before_monitor_loop(self):
        source = (ROOT / "launch_evaluation.bash").read_text()
        rollout = source.index("wait_for_topic /kingfisher/start_navigation")
        benchmark_start = source.rfind('if [ "$benchmark_mode" = "1" ]', 0, rollout)
        monitor_loop = source.index("while ps -p $PY_PID", rollout)
        benchmark_setup = source[benchmark_start:monitor_loop]
        self.assertIn("publish_empty_control /kingfisher/start_navigation 2 30", benchmark_setup)
        monitor_end = source.index("done", monitor_loop)
        benchmark_monitor = source[monitor_loop:monitor_end]
        self.assertNotIn(
            "publish_empty_control /kingfisher/start_navigation 2 30",
            benchmark_monitor,
        )


if __name__ == "__main__":
    unittest.main()
