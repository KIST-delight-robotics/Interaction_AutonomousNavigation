#!/usr/bin/env python3
import json
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# 전역 상태 객체 및 타입들 import
# (실제 패키지 구조에 따라 import 경로 조정 필요)
from state import (
    robot_state,
    RobotState,
    GlobalPhase,
    NavState,
    SpeakState,
    MonitorState,
    ChatMode
)


class FSMNode(Node):
    def __init__(self):
        super().__init__("fsm_node")

        # 이 노드에서 사용할 상태 객체
        # - robot_state 를 그대로 써도 되고,
        #   필요하면 RobotState()로 독립 인스턴스를 써도 된다.
        self.state: RobotState = robot_state

        # LLM이 고른 툴 정보 구독 (agent에서 발행)
        # payload 예: {"intent": "GO", "distance_m": 2.0, "goal": "L8"}
        self.tool_sub = self.create_subscription(
            String,
            "/llm/selected_tool",
            self.tool_callback,
            10,
        )

        # 상태를 외부로 발행
        self.state_pub = self.create_publisher(
            String,
            "/fsm/state",
            10,
        )

        # 실제 주행/정지 명령을 다른 노드로 넘기는 토픽
        self.nav_cmd_pub = self.create_publisher(
            String,
            "/fsm/nav_cmd",  # go/stop 명령용
            10,
        )

        # TTS용 토픽
        self.tts_pub = self.create_publisher(
            String,
            "/fsm/tts",
            10,
        )

        self.get_logger().info(
            f"✅ FSMNode 초기화 완료 (phase={self.state.phase}, "
            f"nav={self.state.nav}, speak={self.state.speak}, monitor={self.state.monitor})"
        )

    # -------------------------
    # 1) LLM 툴 선택 콜백
    # -------------------------
    def tool_callback(self, msg: String):
        self.get_logger().info(f"[FSM] /llm/selected_tool 수신: {msg.data}")

        try:
            spec = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"[FSM] JSON 파싱 실패: {e}")
            return

        intent = spec.get("intent", "NONE")
        distance_m = spec.get("distance_m", None)
        goal = spec.get("goal", None)
        reply_text = spec.get("reply_text", None)
        
        # intent 기반으로 상태 전이
        self.apply_command(intent, distance_m, goal,reply_text)

        # 전이 결과 상태 publish
        self.publish_state()

    # -------------------------
    # 2) 툴별 상태 전이 로직
    # -------------------------

    def apply_command(self, intent: str, distance_m: float | None, goal: str | None, reply_text:str | None = None):
        self.get_logger().info(
            f"[FSM] apply_command 호출: intent={intent}, distance_m={distance_m}, goal={goal}"
        )

        if intent == "GO":
            dist = distance_m if distance_m is not None else 1.0
            self.handle_go(distance_m=dist)

        elif intent == "STOP":
            self.handle_stop()

        elif intent == "SET_GOAL":
            self.handle_set_goal(goal)
        elif intent in ("LAB_CHAT", "GENERAL_CHAT"):
            self.handle_chat(intent, reply_text)
        else:  # "NONE" 또는 기타 값
            self.get_logger().info("[FSM] 제어 명령 아님 (intent=NONE), 상태 변경 없음")

    def handle_go(self, distance_m: float):
        """
        go 선택 시 상태 전이 & 주행 명령 발행
        """
        # START/END에서 go가 오면 ING로 전이
        if self.state.phase in (GlobalPhase.START, GlobalPhase.END):
            self.state.phase = GlobalPhase.ING

        # 하위 상태 전이
        self.state.nav = NavState.ON
        self.state.speak = SpeakState.TALKING  # "출발합니다" TTS라고 가정
        self.state.monitor = MonitorState.ON

        # 실제 nav 명령 토픽 발행 (여기는 high-level command만)
        cmd = {
            "command": "GO",
            "distance_m": distance_m,
        }
        nav_msg = String()
        nav_msg.data = json.dumps(cmd, ensure_ascii=False)
        self.nav_cmd_pub.publish(nav_msg)
        self.get_logger().info(f"[FSM] nav_cmd 발행: {nav_msg.data}")

        # TTS 안내 (옵션)
        tts = String()
        tts.data = f"{distance_m}미터 앞으로 이동할게요."
        self.tts_pub.publish(tts)

    def handle_stop(self):
        """
        stop 선택 시 상태 전이 & 정지 명령 발행
        """
        # phase는 일단 ING 유지, nav만 끄고 speak=TALKING으로 응답했다고 가정
        self.state.nav = NavState.OFF
        self.state.speak = SpeakState.TALKING
        # monitor는 계속 ON으로 유지 (필요시 OFF로 변경 가능)

        cmd = {
            "command": "STOP",
        }
        nav_msg = String()
        nav_msg.data = json.dumps(cmd, ensure_ascii=False)
        self.nav_cmd_pub.publish(nav_msg)
        self.get_logger().info(f"[FSM] nav_cmd 발행: {nav_msg.data}")

        tts = String()
        tts.data = "지금 멈출게요."
        self.tts_pub.publish(tts)

    def handle_set_goal(self, goal: str | None):
        """
        set_goal 선택 시 상태 전이 & TTS 안내 (임시 버전)
        실제 네비게이션 목표 변경 로직은 나중에 붙이면 됨.
        """
        if self.state.phase in (GlobalPhase.START, GlobalPhase.END):
            self.state.phase = GlobalPhase.ING

        self.state.monitor = MonitorState.ON
        self.state.speak = SpeakState.TALKING

        tts = String()
        if goal:
            tts.data = f"목적지를 {goal}로 설정할게요."
        else:
            tts.data = "목적지를 다시 말씀해 주세요."
        self.tts_pub.publish(tts)

        self.get_logger().info(f"[FSM] handle_set_goal 실행: goal={goal}")

    def handle_chat(self, intent: str, reply_text: str | None):
        """
        연구소 관련 / 일반 대화에 대한 상태 전이 & TTS 발행
        """
        if self.state.phase in (GlobalPhase.START, GlobalPhase.END):
            self.state.phase = GlobalPhase.ING

        # speak: 말하는 중으로
        self.state.speak = SpeakState.TALKING
        self.state.monitor = MonitorState.ON

        # chat_mode 설정
        if intent == "LAB_CHAT":
            self.state.chat_mode = ChatMode.LAB
        else:
            self.state.chat_mode = ChatMode.GENERAL

        # TTS 발행
        if reply_text:
            tts_msg = String()
            tts_msg.data = reply_text
            self.tts_pub.publish(tts_msg)
            self.get_logger().info(f"[FSM] chat TTS 발행 ({intent}): {reply_text}")
        else:
            self.get_logger().warn("[FSM] handle_chat: reply_text가 비어 있음")

        # 상태 발행
        self.publish_state()

    # -------------------------
    # 3) 상태 발행
    # -------------------------
    def publish_state(self):
        """
        현재 FSM 상태를 JSON으로 발행
        Enum -> 값(str/int) 으로 변환해서 내보낸다.
        """
        state_dict = {
            "base": self.state.phase.value,
            "nav": int(self.state.nav),
            "speak": int(self.state.speak),
            "monitor": int(self.state.monitor),
        }
        msg = String()
        msg.data = json.dumps(state_dict, ensure_ascii=False)
        self.state_pub.publish(msg)
        self.get_logger().info(f"[FSM] 상태 발행: {msg.data}")


def main(args: Optional[list[str]] = None):
    rclpy.init(args=args)
    node = FSMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("FSMNode 종료 신호 수신 (Ctrl+C)")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
