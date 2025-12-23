cmake_minimum_required(VERSION 3.5)
project(sk_planning_lib)

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
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sk_planning_msgs REQUIRED)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
set(dependencies
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  sk_planning_msgs)

# let the compiler search for headers in the include folder
include_directories(include ${dependencies})

add_library(sk_planning_lib SHARED
                            src/skPng.cpp

                            # Maps
                            src/skMapCell.cpp
                            src/skMapDist.cpp

                            # State
                            src/skGrid2D.cpp

                            # Search Compare
                            src/skSearchCompare.cpp
)
ament_target_dependencies(sk_planning_lib ${dependencies})

# this line to exports the library
ament_export_libraries(sk_planning_lib)

ament_export_include_directories(include)
#target_include_directories(sk_robot_lib PUBLIC
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#  $<INSTALL_INTERFACE:include>)

#add_dependencies(sk_robot_lib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(sk_planning_lib ${catkin_LIBRARIES} png)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS sk_planning_lib
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


