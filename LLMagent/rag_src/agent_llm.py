import rclpy
from rclpy.node import Node
from std_msgs.msg import String

### LangGraph + create_agent
import os
from dotenv import load_dotenv

load_dotenv()
from langchain_teddynote import logging
from langchain_teddynote.messages import stream_response

logging.langsmith("test")
import tiktoken
from langchain.chat_models import init_chat_model

from pydantic import BaseModel
from typing import Literal
from langchain_ollama import ChatOllama
from langchain.agents import create_agent  # ✅ v1에서 쓰는 함수
from langchain.tools import tool  # v1에서 tool 데코레이터

from langchain.agents.middleware import wrap_tool_call, SummarizationMiddleware
from langchain_core.messages import ToolMessage, HumanMessage
from langchain.tools import BaseTool
import json
import re

from chat_backends import run_lab_rag, run_general_chat

##############
# RobotCommand 스키마
##############


class RobotCommand(BaseModel):
    intent: Literal[
        "GO",
        "STOP",
        "SET_GOAL",
        "LAB_CHAT",
        "GENERAL_CHAT",
        "NONE"
    ]
    distance_m: float | None = None
    goal: str | None = None
    reply_text: str | None = None   # 대화용 TTS 내용


##############
# Tool 에러 핸들링 미들웨어
##############


@wrap_tool_call
def handle_tool_errors(request, handler):
    """도구 실행을 시도하고, 실패하면 LLM에게 '실패했다'고 알려줍니다."""
    try:
        return handler(request)
    except Exception as e:
        print(f"DEBUG: 도구 오류 발생! {e}")
        return ToolMessage(
            content=f"도구 사용 중 오류 발생: 입력값을 확인해주세요. (오류: {str(e)})",
            tool_call_id=request.tool_call["id"],
        )


##############
# 도구 정의
##############


@tool
def go(distance_m: float = 1.0) -> dict:
    """
    로봇을 앞으로 이동시키기 위한 '계획'을 세우는 툴.
    실제로 로봇을 움직이지 않고, 의도만 반환한다.

    반환 형식:
    {
      "command": "go",
      "distance_m": <float>
    }
    """
    print(f"[TOOL] plan_go(distance_m={distance_m})")
    return {"command": "go", "distance_m": distance_m}


@tool
def stop() -> dict:
    """
    로봇을 멈추기 위한 '계획'을 세우는 툴.

    반환 형식:
    {
      "command": "stop"
    }
    """
    print("[TOOL] plan_stop()")
    return {"command": "stop"}


@tool
def set_goal(goal: str) -> dict:
    """
    로봇의 `목적지`를 바꾸기 위한 툴.
    목적지는 ['L0','L1','L2'] 중 하나가 될 수 있다.

    반환 형식:
    {
      "command": "set_goal",
      "goal": "<목적지>"
    }
    """
    print(f"[TOOL] set_goal(goal={goal})")
    return {"command": "set_goal", "goal": goal}

@tool
def lab_chat(question: str) -> dict:
    """
    KIST연구소 관련 질문에 답하기 위한 대화용 툴.
    RAG 파이프라인을 사용하여 답변을 생성한다.
    """
    print(f"[TOOL] lab_chat(question={question})")
    answer = run_lab_rag(question)
    return {
        "command": "lab_chat",
        "reply": answer
    }


@tool
def general_chat(question: str) -> dict:
    """
    kist연구소와 상관없는 일반 대화를 위한 툴.
    고성능 LLM을 사용하여 답변을 생성한다.
    """
    print(f"[TOOL] general_chat(question={question})")
    answer = run_general_chat(question)
    return {
        "command": "general_chat",
        "reply": answer
    }


tools = [go, stop, set_goal,lab_chat,general_chat]

##############
# 모델 설정
##############

agent_llm = ChatOllama(model="llama3.1:8b")
#agent_llm = ChatOllama(model="exaone3.5:7.8b")
exa_llm = ChatOllama(model="ingu627/exaone4.0:latest")


##############
# 에이전트 시스템 프롬프트
##############

agent_prompt = """
너는 로봇 제어를 위한 ReAct 스타일 에이전트이다.

사용자의 한국어 명령을 읽고, 필요하면 아래 세 도구 중
하나 또는 여러 개를 순차적으로 호출해서
'로봇을 어떻게 움직일지에 대한 계획(plan)'만 세운다.
실제 로봇 제어는 이 계획을 읽는 다른 ROS 노드(FSMNode)가 수행한다.

[현재 상태 정보]

사용자 입력 앞에는 항상 한 줄짜리 현재 FSM 상태가 붙는다.
형식 예시는 다음과 같다.

[현재상태] base=ING, nav=1, speak=1, monitor=1
사용자 발화: 멈춰봐

각 필드는 다음 의미를 가진다.
- base   : 상위 상태 (START/ING/END)
- nav    : 0이면 정지, 1이면 주행 중
- speak  : 0=음성제어 OFF, 1=명령/대화 대기, 2=TTS로 말하는 중
- monitor: 0=모니터링 OFF, 1=ON

너는 이 상태를 참고해서 불필요한 계획을 피해야 한다.
예를 들어:
- nav=1(이미 주행 중)인데 또 "조금만 더 가봐"라고 하면 go를 다시 써도 되지만,
  단순히 거리를 업데이트하는 go만 한 번 쓰면 된다.
- 이미 충분히 멈춰있는 상황(nav=0)에서 "멈춰"라고 하면 stop을 굳이 여러 번 반복할 필요 없다.

[사용 가능한 도구 목록]

1) go
- 기능: 로봇을 현재 진행 방향 기준으로 앞으로 이동시키는 계획을 세운다.
- 파라미터:
  - distance_m (float, 선택): 전진할 거리(미터).
    예: "한 2미터만 가봐", "조금만 앞으로" 같은 표현을 적당한 수치로 변환해서 넣어라.
    거리가 언급되지 않으면 1.0으로 둔다.

2) stop
- 기능: 로봇을 즉시 멈추는 계획을 세운다.
- 파라미터: 없음.
- 예: "멈춰", "스탑", "거기 서" 등.

3) set_goal
- 기능: 로봇의 목표 위치(목적지)를 ['L0', 'L1', 'L2'] 중 하나로 설정하는 계획을 세운다.
- 파라미터:
  - goal (str): 'L0', 'L1', 'L2' 중 하나.

4) lab_chat
- 기능: 사용자의 질문이 연구소, 연구실, 프로젝트, 실험 내용과 관련된 경우
        RAG 기반 파이프라인으로 정보를 찾아 답변한다.
- 파라미터:
  - question (str): 사용자의 질문 전체 문장.

5) general_chat
- 기능: 일상 대화, 잡담, 연구소와 무관한 내용에 대해 답변한다.
- 파라미터:
  - question (str): 사용자의 질문 전체 문장.

[대화 도구 선택 규칙]
- 사용자의 질문이 연구실/연구소, 프로젝트, 실험 장비, 실험 결과,
  논문, 연구 내용 등과 명확히 관련되어 있다면 lab_chat을 사용하라.
- 그 외 일상 대화, 감정, 날씨, 사적인 고민 등은 general_chat을 사용하라.


[중요 지침]

- 도구를 전혀 쓰지 않아도 된다고 판단되면, 툴을 호출하지 않고 자연어 답변만 해도 된다.
- 하지만 이동/정지/목적지 변경과 관련된 발화라면 반드시 적절한 도구(go/stop/set_goal)를 사용해 계획을 만든다.
- 마지막에 사용된 도구의 출력이 최종적인 로봇 계획으로 사용된다.
"""


print("🔍 ChatOllama model =", getattr(agent_llm, "model", None))
print("🔍 ChatOllama default params =", getattr(agent_llm, "_default_params", None))

agent = create_agent(
    model=agent_llm,
    tools=tools,
    middleware=[handle_tool_errors],
    system_prompt=agent_prompt,
)


class LLMAgentNode(Node):
    """
    LLM 에이전트가 만든 RobotCommand를 ROS 토픽으로 발행하는 노드
    """

    def __init__(self):
        super().__init__("llm_agent_node")

        # RobotCommand를 JSON으로 발행하는 토픽
        self.cmd_pub = self.create_publisher(
            String,
            "/llm/selected_tool",
            10,
        )

        # FSM 상태 캐시
        self.last_fsm_state = {
            "base": "START",
            "nav": 0,
            "speak": 0,
            "monitor": 0,
        }

        # FSM 상태 구독
        self.state_sub = self.create_subscription(
            String,
            "/fsm/state",
            self.fsm_state_callback,
            10,
        )
        self.stt_sub = self.create_subscription(
            String,
            "/user_question",          # 예: STT 노드가 발행하는 텍스트 토픽
            self.stt_callback,
            10,
        )

        self.get_logger().info("✅ LLMAgentNode 초기화 완료")

    def fsm_state_callback(self, msg: String):
        try:
            self.last_fsm_state = json.loads(msg.data)
            self.get_logger().info(
                f"[LLMAgentNode]  🤖 FSM state 업데이트: {self.last_fsm_state}"
            )
        except json.JSONDecodeError as e:
            self.get_logger().error(f"[LLMAgentNode] FSM state JSON 파싱 에러: {e}")
    def stt_callback(self, msg: String):
        """
        STT / 텍스트 토픽으로부터 문장을 받았을 때 호출되는 콜백.
        이 텍스트를 그대로 LLM 에이전트 입력으로 사용한다.
        """
        user_text = msg.data.strip()
        if not user_text:
            return

        self.get_logger().info(f"[LLMAgentNode] 🎤 STT 입력 수신: {user_text}")

        # 🔹 현재 FSM 상태 + user_text를 합쳐서 에이전트 실행
        cmd, final_text, _ = self.run_agent_with_state(user_text)

        # run_agent_with_state() 안에서 이미 publish_command(cmd)를 호출하고 있으므로
        # 여기서는 로그만 찍어줘도 된다.
        self.get_logger().info(f"[LLMAgentNode] 🤖 RobotCommand: {cmd}")
        self.get_logger().info(f"[LLMAgentNode] 🗨️ LLM 응답: {final_text}")

    def publish_command(self, cmd: RobotCommand):
        """
        RobotCommand 객체를 JSON String으로 변환해서 publish
        """
        data = cmd.model_dump()
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.cmd_pub.publish(msg)
        self.get_logger().info(f"[LLMAgentNode] /llm/selected_tool 발행: {msg.data}")

    def run_agent_with_state(self, user_text: str):
        """
        FSM 상태를 사용자 발화 앞에 붙여서 LLM에게 넘기기.
        이렇게 하면 LLM이 현재 nav/speak/monitor 상태를 보고
        go/stop/set_goal 사용 여부를 스스로 판단할 수 있다.
        """
        s = self.last_fsm_state
        state_str = (
            f"[현재상태] base={s['base']}, nav={s['nav']}, "
            f"speak={s['speak']}, monitor={s['monitor']}"
        )
        full_input = state_str + "\n사용자 발화: " + user_text

        cmd, final_text, meta = run_agent(full_input)
        self.publish_command(cmd)
        return cmd, final_text, meta


#################
# Tool 결과 → RobotCommand로 변환
#################


def extract_last_plan(messages) -> RobotCommand:
    """
    messages 리스트에서 마지막 ToolMessage를 찾아서
    RobotCommand로 바꿔준다.
    ToolMessage가 없으면 intent=NONE을 리턴.
    """
    last_tool_msg: ToolMessage | None = None

    for m in reversed(messages):
        if m.type == "tool":
            last_tool_msg = m
            break

    if last_tool_msg is None:
        # 툴을 아예 안 썼으면 제어 명령이 아니라고 보고 NONE
        return RobotCommand(intent="NONE")

    data = last_tool_msg.content  # go/stop/set_goal이 반환한 dict

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            data = {"command": "unknown"}

    cmd = data.get("command")

    if cmd == "go":
        return RobotCommand(
            intent="GO",
            distance_m=data.get("distance_m"),
        )
    elif cmd == "stop":
        return RobotCommand(intent="STOP")
    elif cmd == "set_goal":
        return RobotCommand(
            intent="SET_GOAL",
            goal=data.get("goal"),
        )
    elif cmd == "lab_chat":
        return RobotCommand(
            intent="LAB_CHAT",
            reply_text=data.get("reply")
        )
    elif cmd == "general_chat":
        return RobotCommand(
            intent="GENERAL_CHAT",
            reply_text=data.get("reply")
        )
    else:
        return RobotCommand(intent="NONE")



def run_agent(user_text: str):
    state = agent.invoke({"messages": [HumanMessage(content=user_text)]})
    messages = state["messages"]

    print("\n========== [DEBUG] 메시지 타임라인 ==========")
    for m in messages:
        print(type(m), " | ", m.type, " | ", getattr(m, "name", None), " | ", m.content)
        if getattr(m, "tool_calls", None):
            print("  ↳ tool_calls:", m.tool_calls)
    print("============================================\n")

    # 1) 툴 기반 계획 → RobotCommand 로 변환
    robot_cmd = extract_last_plan(messages)

    # 마지막 자연어 AI 응답 + usage_metadata 찾기
    final_text = None
    last_ai_usage = None

    for m in reversed(messages):
        if m.type == "ai" and not getattr(m, "tool_calls", None):
            final_text = m.content
            last_ai_usage = getattr(m, "usage_metadata", None)
            break

    if final_text is None:
        last_msg = messages[-1]
        final_text = last_msg.content
        last_ai_usage = getattr(last_msg, "usage_metadata", None)

    return robot_cmd, final_text, last_ai_usage


# 5. 메인 루프: 한 번 테스트 + 인터랙티브 ----------------------------

if __name__ == "__main__":
    rclpy.init()
    node = LLMAgentNode()

    try:
        # 🔹 이제는 /user_question 토픽에서 텍스트를 받기만 하면 되므로
        #     그냥 spin 으로 콜백만 돌려주면 된다.
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("LLMAgentNode 종료 (Ctrl+C)")

    finally:
        node.destroy_node()
        rclpy.shutdown()

