# Rust viewer setup

This crate reserves modules for a native Rust viewer using wgpu (WebGPU) and
winit (window/input). The current binary only prints a setup message.

1. Install stable Rust with [rustup](https://www.rust-lang.org/tools/install).
   On Windows use the default MSVC toolchain and install the Visual Studio
   Build Tools **Desktop development with C++** workload (including a Windows
   SDK) if prompted. On macOS install the command-line tools with
   `xcode-select --install`.
2. Open a fresh terminal and run `rustc --version` and `cargo --version`.
3. In this folder run `cargo check`, then `cargo run`. The first check
   downloads crates and creates `Cargo.lock`; keep the lockfile for the app.

Tomorrow: bring up a resizable blank window and log the chosen GPU/backend
(DX12 on Windows and Metal on macOS are natural starting points). Then add
keyboard/mouse navigation and validate the transform with a simple reference
scene. After the Python export format is agreed, compare one fixed camera
against Python before profiling projection, sorting, and blending separately.
The proposal's demo target is 30 FPS at 640 × 480 (33.3 ms per frame).
