# ROS 2 Jazzy + Rust + ros2-rust (rclrs), building the cooperative-swarm nodes.
#
# The daemon is down on the dev box, so this image is the reproducible build:
#   docker build -f ros2_ws/Dockerfile -t coop-swarm-ros .
#   docker run --rm -it coop-swarm-ros \
#     ros2 launch swarm_bringup supervisor_demo.launch.py
#
# Context is the REPO ROOT (the Rust workspace and the Python layers are needed).
FROM ros:jazzy-ros-base

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# --- toolchain: Rust + rosidl Rust generator deps + numpy for the bridge ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git python3-pip python3-numpy python3-colcon-common-extensions \
        libclang-dev clang cmake libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && echo 'source $HOME/.cargo/env' >> /root/.bashrc
ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo install cargo-ament-build
RUN pip install --break-system-packages git+https://github.com/ros2-rust/ros2_rust.git#subdirectory=colcon_ros_cargo || \
    pip install --break-system-packages colcon-ros-cargo

WORKDIR /workspace/coop-swarm

# --- ros2-rust (rclrs) sources, cloned into the workspace src/ ---
RUN mkdir -p ros2_ws/src && cd ros2_ws/src \
    && git clone https://github.com/ros2-rust/ros2_rust.git \
    && vcs import . < ros2_rust/ros2_rust_jazzy.repos || true

# --- our sources (repo root is the build context) ---
COPY rust/           ./rust/
COPY ros2_ws/src/    ./ros2_ws/src/
COPY src/planning/layer3_vlm_planner.py data/detector.onnx data/trajectory.npy ./
COPY data/layer2_isam2_results.npz ./
# onnxruntime dylib for the perception node's `load-dynamic` ort backend
RUN pip download onnxruntime -d /tmp/ort --no-deps \
    && cd /tmp/ort && python3 -c "import zipfile,glob; zipfile.ZipFile(glob.glob('*.whl')[0]).extractall('x')" \
    && cp /tmp/ort/x/onnxruntime/capi/libonnxruntime*.so* /usr/local/lib/ \
    && ln -sf /usr/local/lib/libonnxruntime.so.* /usr/local/lib/libonnxruntime.so || true
ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so

# --- build the ROS 2 workspace (messages + rclrs nodes + rclpy bringup) ---
# TWO PASSES on purpose. colcon's merged setup util omits our custom message
# package (swarm_msgs) from `install/setup.bash`, so a single pass leaves
# install/swarm_msgs OFF AMENT_PREFIX_PATH and colcon-ros-cargo can't discover
# the generated swarm_msgs Rust crate ("no matching package named swarm_msgs").
# Pass 1 builds the messages + rclrs; then we force EVERY built prefix onto
# AMENT_PREFIX_PATH so the Rust bindings are discoverable; pass 2 builds the
# rclrs nodes and the rclpy bringup.
RUN source /opt/ros/jazzy/setup.bash \
    && cd ros2_ws \
    && colcon build --packages-up-to swarm_msgs rclrs \
       --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && source install/setup.bash \
    && export AMENT_PREFIX_PATH="$(ls -d $PWD/install/*/ | sed 's:/$::' | tr '\n' ':')$AMENT_PREFIX_PATH" \
    && colcon build --packages-up-to swarm_bringup swarm_supervisor_node swarm_perception_node \
       --cmake-args -DCMAKE_BUILD_TYPE=Release

# convenience entrypoint sources both overlays
RUN echo 'source /opt/ros/jazzy/setup.bash && source /workspace/coop-swarm/ros2_ws/install/setup.bash' \
    >> /root/.bashrc
COPY ros2_ws/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
