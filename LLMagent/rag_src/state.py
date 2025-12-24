# state.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum


# -----------------------------
# 전역 상태 정의 (공유 데이터 구조)
# -----------------------------


class GlobalPhase(str, Enum):
    """상위 라이프사이클 상태: START / ING / END"""
    START = "START"
    ING = "ING"
    END = "END"


class NavState(IntEnum):
    OFF = 0
    ON = 1


class SpeakState(IntEnum):
    OFF = 0       # 제어 단어 X
    LISTEN = 1    # 벤벤/명령 듣는 상태
    TALKING = 2   # TTS 중


class MonitorState(IntEnum):
    OFF = 0
    ON = 1


class ChatMode(str, Enum):
    NONE = "NONE"
    LAB = "KIST"         # 연구실/RAG 대화
    GENERAL = "GENERAL" # 일반 잡담/스몰톡


@dataclass
class RobotState:
    """
    전체 FSM에서 사용하는 공통 상태 컨테이너
    - phase  : 상위 라이프사이클 (START / ING / END)
    - nav    : 주행 on/off
    - speak  : 음성/대화 상태
    - monitor: 모니터링 on/off
    """
    phase: GlobalPhase = GlobalPhase.START
    nav: NavState = NavState.OFF
    speak: SpeakState = SpeakState.OFF
    monitor: MonitorState = MonitorState.OFF
    chat_mode: ChatMode = ChatMode.NONE   

# 모듈 전역에서 쓸 수 있는 기본 상태 객체
robot_state = RobotState()


# -----------------------------
# OO-style 상위 상태 (원래 코드 유지하되 RobotState 사용)
# -----------------------------


class BaseState(ABC):
    def __init__(self, fsm, node):
        self.fsm = fsm   # 상태 전환 관리자
        self.node = node # ROS 노드 (logger, TTS 등)

    @abstractmethod
    def on_enter(self):
        """상태 진입 시 1회 호출"""
        pass

    @abstractmethod
    def on_update(self):
        """주기적으로 호출 (예: timer에서)"""
        pass

    @abstractmethod
    def on_exit(self):
        """상태 종료 시 1회 호출"""
        pass


class StartState(BaseState):
    def on_enter(self):
        self.node.get_logger().info("[STATE] Enter: START")

        # 전역 RobotState 초기화
        robot_state.phase = GlobalPhase.START
        robot_state.nav = NavState.OFF
        robot_state.speak = SpeakState.OFF
        robot_state.monitor = MonitorState.OFF

        # 세션 플래그 초기화
        self.node.session_started = False

        # 인사 멘트 (원하면)
        if hasattr(self.node, "ask_tts"):
            self.node.ask_tts("안녕하세요, 벤벤이에요. 준비가 되면 말을 걸어주세요.")

    def on_update(self):
        # 예: agent가 "SESSION_START" 같은 tool을 선택하면
        # node.session_started 를 True로 바꾸고, 그때 ING로 진입
        if getattr(self.node, "session_started", False):
            self.fsm.change_state(self.node.ing_state)

    def on_exit(self):
        self.node.get_logger().info("[STATE] Exit: START")


class IngState(BaseState):
    def on_enter(self):
        self.node.get_logger().info("[STATE] Enter: ING")

        # ING 진입 시 기본 모드: 주행 꺼짐, 말은 듣기 모드, 모니터링 ON
        robot_state.phase = GlobalPhase.ING
        robot_state.nav = NavState.OFF
        robot_state.speak = SpeakState.LISTEN  # 트리거 단어 대기
        robot_state.monitor = MonitorState.ON

    def on_update(self):
        # 여기서는 매 주기마다 robot_state를 보고 필요한 행동을 관리
        # 예: speak == TALKING 이면 TTS 재생 중, 끝나면 LISTEN으로 돌린다든지.
        if hasattr(self.node, "update_modes"):
            self.node.update_modes()

        # 세션 종료 조건이 만족되면 END로 전환
        if getattr(self.node, "session_ended", False):
            self.fsm.change_state(self.node.end_state)

    def on_exit(self):
        self.node.get_logger().info("[STATE] Exit: ING")


class EndState(BaseState):
    def on_enter(self):
        self.node.get_logger().info("[STATE] Enter: END")

        robot_state.phase = GlobalPhase.END
        robot_state.nav = NavState.OFF
        robot_state.speak = SpeakState.OFF
        robot_state.monitor = MonitorState.OFF

        if hasattr(self.node, "ask_tts"):
            self.node.ask_tts("오늘 도와드려서 기뻤어요. 다음에 또 봐요.")

    def on_update(self):
        # 필요하면 여기서 노드를 종료하거나 START로 되돌릴 수도 있음
        pass

    def on_exit(self):
        self.node.get_logger().info("[STATE] Exit: END")


class FSM:
    """상위 상태(START/ING/END)를 관리하는 간단한 FSM 래퍼"""

    def __init__(self, initial_state: BaseState):
        self.current_state = initial_state
        self.current_state.on_enter()

    def update(self):
        """주기적으로 호출해서 현재 상태의 on_update 실행"""
        self.current_state.on_update()

    def change_state(self, new_state: BaseState):
        """상태 전환"""
        if type(self.current_state) == type(new_state):
            return
        self.current_state.on_exit()
        self.current_state = new_state
        self.current_state.on_enter()
