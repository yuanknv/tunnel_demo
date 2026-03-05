#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="${PIXI_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"

# Default parameters
WIDTH=1920
HEIGHT=1080
BACKEND="cuda"
PUBLISH_RATE=1

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --rate)
            PUBLISH_RATE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --width WIDTH         Image width (default: 3840)"
            echo "  --height HEIGHT       Image height (default: 2160)"
            echo "  --backend BACKEND     cuda or cpu (default: cuda)"
            echo "  --rate RATE_MS        Publish rate in ms (default: 1)"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate backend
if [[ "$BACKEND" != "cuda" && "$BACKEND" != "cpu" ]]; then
    echo "Error: backend must be 'cuda' or 'cpu', got: $BACKEND"
    exit 1
fi

USE_CUDA="true"
if [[ "$BACKEND" == "cpu" ]]; then
    USE_CUDA="false"
fi

. "$WS_ROOT/build/tunnel_demo/colcon_command_prefix_build.sh"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
# Ensure the build and install lib directories are visible to the loader so
# plugin libraries (component .so files) can be found when running build
# executables from the build directory.
export LD_LIBRARY_PATH="$WS_ROOT/build/tunnel_demo:$WS_ROOT/install/tunnel_demo/lib:$WS_ROOT/.pixi/envs/default/libtorch/lib:${LD_LIBRARY_PATH:-}"

ZENOHD="$WS_ROOT/install/rmw_zenoh_cpp/lib/rmw_zenoh_cpp/rmw_zenohd"
RENDERER="$WS_ROOT/build/tunnel_demo/tunnel_renderer_node"
DISPLAY_NODE="$WS_ROOT/build/tunnel_demo/tunnel_display_node"

cleanup() {
    kill $ZENOH_PID $RENDERER_PID $DISPLAY_PID 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "Starting zenoh router..."
$ZENOHD &
ZENOH_PID=$!
sleep 1

echo "Starting tunnel renderer (publisher)..."
echo "  Resolution: ${WIDTH}x${HEIGHT}"
echo "  Backend: $BACKEND"
echo "  Rate: ${PUBLISH_RATE}ms"
$RENDERER --ros-args \
    -p image_width:=$WIDTH \
    -p image_height:=$HEIGHT \
    -p use_cuda:=$USE_CUDA \
    -p publish_rate_ms:=$PUBLISH_RATE &
RENDERER_PID=$!

echo "Starting tunnel display (subscriber)..."
$DISPLAY_NODE &
DISPLAY_PID=$!

echo "All nodes running. Press Ctrl+C to stop."
wait
