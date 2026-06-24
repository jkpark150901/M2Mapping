# Test And Run Commands

Commands assume the repository root is the current directory.

## Configure And Build

Local host using `.venv-vis` LibTorch:

```bash
cmake --preset local-venv -S standalone_mapping
cmake --build standalone_mapping/build -j2
```

Manual Torch path:

```bash
cmake -S standalone_mapping -B standalone_mapping/build \
  -DBUILD_COMPONENT_TESTS=ON \
  -DBUILD_UNIT_TESTS=ON \
  -DSTANDALONE_CUDA_ARCH_LIST=8.6 \
  -DSTANDALONE_USE_PREBUILT_ROG_MAP=OFF \
  -DPREBUILT_ROG_MAP_LIB=/home/dev/.local/share/enroot/m2mapping/m2mapping_ws/devel/lib/librog_map_cuda.so \
  -DTorch_DIR=/home/dev/jkpark/M2Mapping/.venv-vis/lib/python3.10/site-packages/torch/share/cmake/Torch
cmake --build standalone_mapping/build -j2
```

Inside enroot, if LibTorch is available at `/usr/local/lib/libtorch`:

```bash
cmake --preset enroot-libtorch -S standalone_mapping
cmake --build standalone_mapping/build-enroot -j2
```

The enroot preset builds `submodules/ROG-Map` from source. The local preset uses
the prebuilt ROG-Map shared library only as a host-side workaround.

## Unit Tests

Run the lightweight C++ unit test binary directly:

```bash
standalone_mapping/build/tests/unit_tests/mapping_unit_tests
```

Run through CTest:

```bash
ctest --test-dir standalone_mapping/build --output-on-failure
```

Current unit tests cover:

- `MappingSystem` config storage
- dataset pointer ownership
- current `FolderDataset` placeholder contract
- visualization stage directory creation
- placeholder `Trainer` and `Renderer` calls
- `TeacherGeometry` query result shape contract

## GPU Occ Grid Generation Test

This test requires a CUDA-capable device because it calls ROG-Map.

Configure with GPU integration tests enabled:

```bash
cmake --preset local-venv -S standalone_mapping -DBUILD_GPU_TESTS=ON
cmake --build standalone_mapping/build -j2
```

Run only the occ grid generation test with the dataset/config chosen at test
time:

```bash
OCC_GRID_TEST_DATA=data/iae_map \
OCC_GRID_TEST_CONFIG=standalone_mapping/config/scenes/iae_map.yaml \
OCC_GRID_TEST_RUN=/tmp/standalone_occ_test \
ctest --test-dir standalone_mapping/build \
  -R occ_grid_generation \
  --output-on-failure
```

Equivalent direct command:

```bash
standalone_mapping/build/tests/component_tests/mapping_component_tests occ_grid \
  --config standalone_mapping/config/scenes/example.yaml \
  --data standalone_mapping/examples/tiny_occ_dataset \
  --run /tmp/standalone_occ_smoke
```

Expected outputs:

```text
/tmp/standalone_occ_test/as_prior.ply
/tmp/standalone_occ_test/visualization/occupancy_grid/snapshot.ply
/tmp/standalone_occ_test/visualization/occupancy_grid/snapshot_summary.txt
```

## Main CLI Smoke Tests

These commands exercise the standalone command surface. `train`, `eval`, and
`render` currently call placeholder implementations. `occmap` now expects a
folder dataset with ASCII PLY depth clouds.

```bash
standalone_mapping/build/mapping_app occmap \
  --config standalone_mapping/config/scenes/example.yaml \
  --data standalone_mapping/examples/tiny_occ_dataset \
  --run /tmp/standalone_occ_smoke

standalone_mapping/build/mapping_app train \
  --config standalone_mapping/config/scenes/example.yaml \
  --data /tmp/example_dataset \
  --iters 2

standalone_mapping/build/mapping_app eval \
  --run output/example

standalone_mapping/build/mapping_app render \
  --run output/example \
  --poses poses.txt
```

## Component Test Smoke Tests

These commands fix the future component-test interface for the ROS-free project.
`occ_grid` calls the same ROG occupancy builder as `mapping_app occmap`. The
remaining modes currently print parsed arguments and return success.

```bash
standalone_mapping/build/tests/component_tests/mapping_component_tests occ_grid \
  --config standalone_mapping/config/scenes/example.yaml \
  --data standalone_mapping/examples/tiny_occ_dataset \
  --run /tmp/standalone_occ_component

standalone_mapping/build/tests/component_tests/mapping_component_tests ray_trace \
  --run output/example \
  --num-rays 4096

standalone_mapping/build/tests/component_tests/mapping_component_tests lidar_sampling \
  --run output/example \
  --num-rays 4096

standalone_mapping/build/tests/component_tests/mapping_component_tests sdf_alpha \
  --run output/example \
  --num-rays 4096

standalone_mapping/build/tests/component_tests/mapping_component_tests teacher_query \
  --run output/example \
  --teacher standalone_mapping/config/teacher/floorplan.yaml \
  --num-rays 4096
```

## Next Test Implementation Targets

Replace placeholders in this order:

1. `FolderDataset` loads `config.yaml`, poses, images, and depth/point tensors.
2. `occ_grid` calls the moved `OccupancyBuilder` and verifies `as_prior.ply`.
3. `ray_trace` loads a run folder and calls `tracer::render_ray`.
4. `lidar_sampling` calls `LocalMap::sample`.
5. `sdf_alpha` saves `sdf`, `isigma`, `delta`, `alpha`, and weights.
6. `teacher_query` queries teacher SDF and saves teacher/student tensors.
