#!/bin/bash

# A script to find and kill a process running on a specific port.
# Usage: ./kill_port.sh [port_number]
# If no port number is provided, it defaults to 6006.

# Set the port to the first argument, or default to 6006
PORT=${1:-6006}

echo "Searching for process on port $PORT..."

# Find the PID using the 'ss' command and parse the output
PID=$(ss -lptn "sport = :$PORT" | grep -o 'pid=[0-9]*' | cut -d'=' -f2)

# Check if a PID was found
if [ -z "$PID" ]; then
  echo "No process found running on port $PORT."
else
  echo "Found process with PID: $PID. Terminating..."
  # Kill the process forcefully
  kill -9 "$PID"
  echo "Process $PID killed. Port $PORT should now be free."
fi