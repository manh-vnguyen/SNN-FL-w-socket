#!/bin/bash

# Paths to the files you want to monitor
FILE_PATH_SERVER="/home/flwr_srvr/test_socket_server/server_w_socket.py"
FILE_PATH_CLIENT="/home/flwr_srvr/test_socket_server/client_w_socket.py"

# Initial hash of the files
LAST_HASH_SERVER=$(md5sum "$FILE_PATH_SERVER" | awk '{ print $1 }')
LAST_HASH_CLIENT=$(md5sum "$FILE_PATH_CLIENT" | awk '{ print $1 }')

restart_service() {
    local file_path=$1
    local session_name=""
    local command=""

    if [ "$file_path" == "$FILE_PATH_SERVER" ]; then
        session_name="socket_server"
        command="python3 server_w_socket.py"
    elif [ "$file_path" == "$FILE_PATH_CLIENT" ]; then
        session_name="socket_client"
        command="python3 client_w_socket.py"
    else
        echo "Invalid file path: $file_path"
        return
    fi

    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "Session $session_name already exists."
    else
        echo "Creating new tmux session $session_name."
        tmux new-session -d -s "$session_name"
    fi

    echo "Sending command to $session_name."
    tmux send-keys -t "$session_name" C-c C-c " $command" C-m 
}

on_file_content_changed() {
    local file_path=$1
    echo "Content of file $file_path has changed."
    restart_service $file_path
}

# Monitoring file modifications using inotifywait in a loop
while true; do
  inotifywait -e close_write "$FILE_PATH_SERVER" "$FILE_PATH_CLIENT"
  CURRENT_HASH_SERVER=$(md5sum "$FILE_PATH_SERVER" | awk '{ print $1 }')
  CURRENT_HASH_CLIENT=$(md5sum "$FILE_PATH_CLIENT" | awk '{ print $1 }')

  if [ "$LAST_HASH_SERVER" != "$CURRENT_HASH_SERVER" ]; then
    on_file_content_changed "$FILE_PATH_SERVER"
    LAST_HASH_SERVER=$CURRENT_HASH_SERVER
  fi
  
  if [ "$LAST_HASH_CLIENT" != "$CURRENT_HASH_CLIENT" ]; then
    on_file_content_changed "$FILE_PATH_CLIENT"
    LAST_HASH_CLIENT=$CURRENT_HASH_CLIENT
  fi
done

