#!/usr/bin/env bash
# run_tmux.sh
SESSION=gps_controller
#CMD="uvicorn gps_server:app --host 0.0.0.0 --port 5000"
CMD="uvicorn planner_server:app --host 0.0.0.0 --port 5000"

# 세션이 없으면 새로 만들고 명령 실행
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    # 빈 세션(기본 bash) 생성
    tmux new-session -d -s "$SESSION" -n main

    # 셸에 uvicorn 명령 입력 + Enter
    # tmux send-keys -t "$SESSION":0 "conda activate gps-train" C-m
    tmux send-keys -t "$SESSION":0 "$CMD" C-m
fi

# 세션에 붙기
tmux attach -t "$SESSION"

