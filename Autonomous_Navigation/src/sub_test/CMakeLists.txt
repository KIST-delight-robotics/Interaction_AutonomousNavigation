cmake_minimum_required(VERSION 3.5)
project(sub_test)

# Default to C++14
#if(NOT CMAKE_CXX_STANDARD)
#  set(CMAKE_CXX_STANDARD 14)
#endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(aruco_opencv_msgs REQUIRED)
find_package(sk_robot_lib REQUIRED)# COMPONENTS sk_robot_lib)

set(dependencies
  rclcpp
  std_msgs
  geometry_msgs
  aruco_opencv_msgs
  sensor_msgs
  sk_robot_lib)

include_directories(include)

# files to build
add_executable(run_test src/run_test.cpp)
ament_target_dependencies(run_test ${dependencies})
add_executable(run_scan2grid src/run_scan2grid.cpp)
ament_target_dependencies(run_scan2grid ${dependencies})
#target_link_libraries(kibotOmni sk_robot_lib)
#target_include_directories(kibotOmni PUBLIC
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#  $<INSTALL_INTERFACE:include>
#  sk_robot_lib_EXPORTS)

#add_message_files(
#  FILES
#)

#generate_messages(
#  DEPENDENCIES
#  std_msgs
#)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS run_test run_scan2grid
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})

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
