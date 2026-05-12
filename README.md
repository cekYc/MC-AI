# 🤖 mcAI: Minecraft Swarm Intelligence

mcAI is a high-performance, multi-agent reinforcement learning framework for Minecraft. It leverages a high-concurrency **Rust** core for bot orchestration and a **Python** brain for deep learning and swarm intelligence.

## 🏗️ Architecture

- **`rust_core`**: Built with [Azalea](https://github.com/azalea-rs/azalea), this module handles Minecraft protocol implementation, physics, and low-level bot actions. It communicates with the brain via gRPC.
- **`python_brain`**: Powered by [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) and [PyTorch](https://pytorch.org/), this module implements multi-agent reinforcement learning algorithms to train and execute swarm behaviors.
- **`shared`**: Contains the [Protocol Buffers](https://protobuf.dev/) definitions (`swarm.proto`) that facilitate communication between Rust and Python.

## 🚀 Features

- **Multi-Agent Coordination**: Designed to handle swarms of Minecraft bots simultaneously.
- **High Performance**: Rust core ensures minimal latency and low resource usage for protocol handling.
- **Deep RL Integration**: Seamlessly connect complex learning models to game environments.
- **Synchronized Learning**: State injection logic optimized for swarm exploration rewards.

## 🛠️ Getting Started

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (Edition 2024)
- [Python 3.10+](https://www.python.org/downloads/)
- [Protobuf Compiler](https://grpc.io/docs/protoc-installation/) (optional, included in build script)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cekYc/MC-AI.git
   cd MC-AI
   ```

2. **Setup the Python Brain:**
   ```bash
   cd python_brain
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **(Optional) Regenerate gRPC code:**
   If you modify `shared/swarm.proto`, run:
   ```bash
   python -m grpc_tools.protoc -I../shared --python_out=. --grpc_python_out=. ../shared/swarm.proto
   ```

4. **Build the Rust Core:**
   ```bash
   cd ../rust_core
   cargo build --release
   ```

## 🎮 Usage

1. **Start the Brain Server:**
   ```bash
   cd python_brain
   python server.py
   ```

2. **Launch the Bots:**
   ```bash
   cd rust_core
   cargo run --release
   ```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Built with ceky for the Minecraft AI community.*
