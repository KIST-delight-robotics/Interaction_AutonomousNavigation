/**********************************************************************
 * DWA → Go2 Command Bridge + Watchdog
 *  - Subscribe: /cmd (geometry_msgs::msg::Twist)
 *  - Publish: Go2 SportClient API 호출
 *  - Watchdog: 일정 시간 /cmd 미수신 시 StopMove
 ***********************************************************************/

#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include "common/ros2_sport_client.h"
#include "unitree_go/msg/sport_mode_state.hpp"

#define TOPIC_HIGHSTATE "lf/sportmodestate"

using namespace std::chrono_literals;

class Dwa2Go2Node : public rclcpp::Node {
public:
    Dwa2Go2Node() : Node("dwa2go2_node"), sport_client_(this) {
        // Subscribe to /cmd (Twist)
        sub_cmd_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd", 10,
            std::bind(&Dwa2Go2Node::cmdCallback, this, std::placeholders::_1));

        // Subscribe to Go2 HighState (optional, for feedback/debug)
        sub_state_ = this->create_subscription<unitree_go::msg::SportModeState>(
            TOPIC_HIGHSTATE, 1,
            [this](const unitree_go::msg::SportModeState::SharedPtr msg) {
                state_ = *msg;
            });

        // 워치독 체크
        timer_ = this->create_wall_timer(
            100ms, std::bind(&Dwa2Go2Node::watchdogCheck, this));

        last_cmd_time_ = this->now();  // 초기화
        RCLCPP_INFO(this->get_logger(), "DWA→Go2 bridge node with watchdog started.");
    }

private:
    void cmdCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        auto now_time = this->now();

        // Hz 계산 (직전 /cmd와의 시간 간격)
        double dt = (now_time - last_cmd_time_).seconds();
        if (dt > 0.0) {
            cmd_freq_ = 1.0 / dt;
        }

        // 마지막 cmd 수신 시각 갱신
        last_cmd_time_ = now_time;

        // Go2 API 호출
        sport_client_.Move(req_, msg->linear.x, msg->linear.y, msg->angular.z);

        RCLCPP_INFO(this->get_logger(),
                    "Cmd Hz: %.1f | Send to Go2: vx=%.2f, vy=%.2f, wz=%.2f",
                    cmd_freq_, msg->linear.x, msg->linear.y, msg->angular.z);
    }

    void watchdogCheck() {
        auto elapsed = (this->now() - last_cmd_time_).seconds();
        if (elapsed > 0.3) {
            sport_client_.StopMove(req_);
            RCLCPP_WARN(this->get_logger(),
                        "No /cmd for %.2f s → StopMove sent", elapsed);
        }
    }

    // Members
    SportClient sport_client_;
    unitree_api::msg::Request req_;
    unitree_go::msg::SportModeState state_;

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_;
    rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr sub_state_;
    rclcpp::TimerBase::SharedPtr timer_;

    rclcpp::Time last_cmd_time_;
    double cmd_freq_ = 0.0;


};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Dwa2Go2Node>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

