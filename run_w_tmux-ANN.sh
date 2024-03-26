#!/bin/bash

# Initialize variables
ACTION=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        start|halt|cleanup|cleantmux) ACTION="$1"; shift ;;
        --train_log_prefix) TRAIN_LOG_PREFIX="$2"; shift 2 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Function to start a new tmux session or send commands to an existing one
start_or_send_command() {
  local session_name="$1"
  local command="$2"

  if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "Session $session_name already exists."
  else
    echo "Creating new tmux session $session_name."
    tmux new-session -d -s "$session_name"
  fi

  echo "Sending command to $session_name."
  tmux send-keys -t "$session_name" "$command" C-m
}

# Function to halt the command in a tmux session
halt_command_in_session() {
  local session_name="$1"

  if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "Halting command in tmux session $session_name."
    tmux send-keys -t "$session_name" C-c
  else
    echo "Session $session_name does not exist."
  fi
}

# Function to clean up all log files with the given prefix
cleanup_logs() {
    local prefix="$1"
    echo "Cleaning up log files in plot_data/ with prefix $prefix..."
    rm -f plot_data/"${prefix}"*.out
    echo "Cleanup completed."
}

# Function to clean up all tmux sessions
clean_tmux_sessions() {
    tmux list-sessions | grep -E 'server_session|client_session_[0-9]+' | cut -d: -f1 | while read session_name; do
        tmux kill-session -t "$session_name"
        echo "Killed tmux session $session_name"
    done
}

NUM_CLIENTS=7

case "$ACTION" in
  start)
    server_command="echo Starting server; python3 server_w_socket-ANN.py"
    start_or_send_command "server_session" "$server_command"
    sleep 10
    for i in $(seq 0 $NUM_CLIENTS); do
        client_command="echo Starting client $i; python3 client_w_socket-ANN.py --node-id $i"
        start_or_send_command "client_session_$i" "$client_command"
    done
    ;;
  halt)
    halt_command_in_session "server_session"
    for i in $(seq 0 $NUM_CLIENTS); do
      halt_command_in_session "client_session_$i"
    done
    ;;
  cleanup)
    if [ -z "$TRAIN_LOG_PREFIX" ]; then
      echo "train_log_prefix is required for cleanup action."
      exit 1
    fi
    cleanup_logs "$TRAIN_LOG_PREFIX"
    ;;
  cleantmux)
    clean_tmux_sessions
    ;;
  *)
    echo "Unknown action: $ACTION"
    exit 1
    ;;
esac
