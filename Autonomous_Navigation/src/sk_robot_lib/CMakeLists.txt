cmake_minimum_required(VERSION 3.5)
project(sk_robot_lib)

## Default to C99
#if(NOT CMAKE_C_STANDARD)
#  set(CMAKE_C_STANDARD 99)
#endif()

# Default to C++14
#if(NOT CMAKE_CXX_STANDARD)
#  set(CMAKE_CXX_STANDARD 14)
#endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(aruco_opencv_msgs REQUIRED)
find_package(dynamixel_sdk REQUIRED)
find_package(sk_robot_msgs REQUIRED)
#find_package(catkin REQUIRED COMPONENTS
#  roslib
#  sensor_msgs
#  cv_bridge
#  message_generation
#)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
set(dependencies
  rclcpp
  std_msgs
  tf2
  tf2_ros
  geometry_msgs
  sensor_msgs
  nav_msgs
  aruco_opencv_msgs
  dynamixel_sdk
  sk_robot_msgs)

# let the compiler search for headers in the include folder
include_directories(include ${dependencies})# sk_robot_msgs)

add_library(sk_robot_lib SHARED
                            ### Common Functions ###
                            src/skRobotBasicFunctions.cpp
                            src/skAvgFilter.cpp
                            src/skSerial.cpp
                            src/skCSE3.cpp

                            ### Subscribers ###
                            src/skSubMsg.cpp
                            src/skSubF64MA.cpp
                            src/skSubTwist.cpp
                            src/skSubPose2D.cpp
                            src/skSubImu.cpp
                            src/skSubLaserScan.cpp
                            src/skSubArucoDetection.cpp
                            src/skSubRobotState.cpp
                            src/skSubJoy.cpp
                            src/skSubJoyMobile.cpp
                            src/skSubQuaternionStamped.cpp
                            src/skSubNavSatFix.cpp
                            src/skSubMag.cpp
                            src/skSubOccupancyGrid.cpp

                            ### Servoes ###
                            src/skServo.cpp
                            src/skServoDynamixel.cpp
                            src/skServoMD400T.cpp
                            src/skServoMD400Trio.cpp
                            src/skServoOdrive.cpp
                            src/skServoWheelTec.cpp
                            src/skServoMimic.cpp

                            ### Kinematics/Dynamics ###
                            src/skKinematicsMobile.cpp
                            src/skKinematicsMobile2WDD.cpp
                            src/skKinematicsMobile3WhOmni.cpp
                            src/skDynamicsMobile3WhOmni.cpp
                            src/skKinematicsMobilePass.cpp

                            ### Robot ###
                            src/skRobot.cpp

                            ### Actions ###
                            src/skAction.cpp
                            src/skActionDash.cpp
                            src/skActionWallFollow.cpp
                            src/skActionBatteryMeasure.cpp
                            src/skActionDriveGPS.cpp

                            ### Controllers ###
                            src/skCDWController.cpp

                            ### Collision Avoidance ###
                            src/skDynamicObstacleAvoidance.cpp
                            src/skDynamicWindowApproach.cpp
                            
#                            src/skSubPoseStamped.cpp
#                            src/skSubAlvarMarker.cpp
#                            src/skCCollisionMap.cpp
#                            src/skCLaserScannerStorage.cpp
#                            src/skCLaserScannerContainer.cpp
#                            src/skCDWController.cpp
#                            src/skGazeboPoseExt.cpp
#                            src/skCMDServo.cpp
#                            src/skCMDTrioMotionTCP.cpp
#                            src/skCMDDynamixel.cpp
#                            src/skSubRobotState.cpp
#                            src/skCImage.cpp
#                            src/skCustomServo.cpp
#                            src/skCLedControl.cpp
#                            src/skDynamixelServo.cpp
#                            src/skCPoseFilter.cpp
)
ament_target_dependencies(sk_robot_lib ${dependencies})# sk_robot_msgs)

# this line to exports the library
ament_export_libraries(sk_robot_lib)
#ament_export_targets(export_sk_robot_lib HAS_LIBRARY_TARGET)
#ament_export_dependencies(rclcpp std_msgs geometry_msgs nav_msgs dynamixel_sdk sk_robot_msgs)
ament_export_dependencies(${dependencies})

ament_export_include_directories(include)# ${dependencies})# sk_robot_lib)
#target_include_directories(sk_robot_lib PUBLIC
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#  $<INSTALL_INTERFACE:include>)

#add_dependencies(sk_robot_lib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
#target_link_libraries(sk_robot_lib ${catkin_LIBRARIES} odrive_cpp_sdk)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS sk_robot_lib
  #EXPORT export_sk_robot_lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)
  #DESTINATION lib/${PROJECT_NAME})



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
