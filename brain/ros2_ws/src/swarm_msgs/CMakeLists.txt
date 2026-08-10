cmake_minimum_required(VERSION 3.8)
project(swarm_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BearingObservation.msg"
  "msg/SwarmEstimate.msg"
  "msg/Assignment.msg"
  "msg/MissionPlan.msg"
  "msg/PlanDecision.msg"
)

ament_package()
