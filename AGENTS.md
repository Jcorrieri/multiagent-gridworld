# Repository Guide

## Coding Conventions

- Prefer modular, reusable functions and components.
- Keep lines under 100 characters.
- Keep functions under 90 lines.
- Prefer dependency-free changes where reasonable. Ask before installing dependencies.
- Preserve unrelated working-tree changes and generated experiment artifacts.

## Internal Code Structure

### Runtime flow

- `main.py` is the CLI entry point. It loads `config/<name>`, creates the reward scheme,
  registers the RLlib environment, and dispatches to training or testing.
- `train.py` builds and runs PPO, creates versioned experiment directories, saves checkpoints,
  and plots training metrics.
- `test.py` restores an RLlib checkpoint or runs the deterministic baseline, evaluates episodes,
  and writes coverage, makespan, connectivity, and timing CSVs.
- `utils.py` owns CLI flags, environment registration and construction, reward selection,
  plotting, and obstacle-map generation.

### Environment and rewards

- `environment/envs/gridworld.py` defines the PettingZoo `ParallelEnv` used for learning.
  It owns maps, movement, collisions, coverage, communication graphs, observations, and rendering.
- Each agent receives a dictionary observation with `actor` and `critic` tensors. The actor uses
  knowledge shared by its current communication component; the critic receives global state.
- Component maps track discovered obstacles and visited tiles. Maps merge when communication
  components join and are cloned when a component splits.
- `environment/envs/baseline.py` extends the grid world with frontier selection, configuration
  scoring, deadlock detection, and deadlock recovery.
- `environment/rewards.py` contains the `Default`, `Coverage`, `ExplorerMaintainer`, and
  `Components` reward schemes. `utils.make_reward_scheme()` maps config names to these classes.
- `environment/obstacle-mats/{training,testing}/` each contain 50 coordinate-based 25x25 maps.

### Models and RLlib

- `models/rl_wrapper.py` is an old-stack RLlib `TorchModelV2` adapter. It dynamically imports the
  configured architecture from `models/arch/` and exposes policy logits and a value function.
- `models/arch/cnn_2conv2linear.py` is the only checked-in network. It uses a shared convolutional
  encoder with separate actor and critic heads.
- `train.py` registers one shared policy for every robot. It explicitly disables RLModule,
  Learner, EnvRunner v2, and ConnectorV2 behavior through `.api_stack()`.
- Checked-in checkpoints under `experiments/gridworld/v4/saved/` were written by Ray 2.44.1.
  Treat checkpoint compatibility as an explicit migration concern.

### Configuration, outputs, and supporting files

- `config/default` is the active YAML configuration for environment, training, and testing.
- `experiments/<environment>/v*/` contains copied configs, checkpoints, saved algorithms,
  training plots, and test results. These are artifacts, not source modules.
- `visualization.py`, `figures/`, and `final_results/` are offline analysis and plotting assets.
- `README.md` describes installation and usage. `.github/copilot-instructions.md` overlaps with
  this guide and may lag behind source changes.
- There is no dependency manifest or automated test suite. The executable blocks at the bottom
  of the environment modules are manual smoke tests, not unit tests.

## Known Issues and Modernization Priorities

### High priority: environment correctness

- Local vision is a square crop and does not perform line-of-sight checks. Agents discover cells
  and obstacles through walls. Replace this with obstacle-aware visibility and test occlusion at
  walls, corners, map edges, and overlapping fields of view.
- Movement is resolved sequentially in fixed agent order. Outcomes can depend on agent ID when
  robots vacate or contest cells. Compute proposals first, resolve conflicts simultaneously, then
  apply accepted moves.
- Reset initializes every robot with one shared component map and reports connectivity as intact
  without deriving components from the initial communication graph. This becomes incorrect once
  spawn positions are randomized.
- Map loading assumes exactly 50 non-empty `matN` files with valid two-column coordinates.
  Empty, single-coordinate, missing, malformed, or wrong-sized maps are not handled safely.
- Actions are assumed to contain every live agent and a valid integer in the action space.
  Invalid or incomplete action dictionaries fail indirectly.

### High priority: spawning and environment complexity

- Agents spawn deterministically along the bottom rows. The placement arithmetic contains a
  literal `25`, can produce invalid positions for some sizes and agent counts, and depends on map
  generation clearing the bottom three rows.
- Random spawning should use the environment RNG, select unique free cells, define whether the
  initial team must be connected, and handle a fixed base station explicitly.
- `num_maps`, the local-FOV threshold, model dimensions, and parts of spawn logic assume a 25x25
  world even though `size` is configurable.
- Map generation supports only a narrow density mix and clears a fixed spawn strip after its
  connectivity check. More complex maps need explicit complexity controls and post-processing
  validation so free-space reachability remains guaranteed.

### High priority: RLlib and model APIs

- The project uses Ray 2.44.1's old `ModelV2` stack and explicitly disables current RLlib APIs.
  Migrate training and inference together to `RLModule`, `RLModuleSpec`, Learner, EnvRunner, and
  ConnectorV2 APIs before extending the model architecture.
- Replace dynamic file-based model loading with an explicit module or model registry. Model
  configuration should declare observation channels, encoder dimensions, and actor/critic heads.
- The CNN hard-codes a 25x25x4 input and does not derive its shape from the observation space.
  Observation changes or different map sizes therefore fail at the first linear layer.
- Training always requests one GPU even when `main.py` detects CPU-only execution. Resource and
  worker settings should be explicit configuration with safe local defaults.
- Old checkpoints may require an intentional conversion path or may be declared unsupported.
  Do not silently overwrite or treat them as compatible with redesigned observations or models.

### Reproducibility and evaluation

- `reset(seed=...)` replaces the environment RNG but does not rebuild the pending map permutation,
  so the next map is not fully determined by the supplied reset seed.
- Baseline search and deadlock recovery create fresh unseeded NumPy generators instead of using
  the environment RNG. Seeded baseline evaluations are not reproducible.
- `training.num_episodes` is converted to iterations using `max_steps`; early episode termination
  means it is not an exact episode budget.
- A disconnection still active at episode termination is emitted without an end step or duration.
- Testing and training do not consistently close RLlib algorithms or Ray resources on failure.
- Archived configuration is stale: `experiments/gridworld/v4/config` uses `model_path`, while the
  evaluator reads `model_version`; root `v5` references an architecture that is not checked in.

### Maintainability and validation gaps

- Several environment and rendering functions exceed 90 lines, and many source lines exceed
  100 characters. Refactors should separate state transitions, visibility, observations, and UI.
- `gridworld.py` and `baseline.py` duplicate substantial rendering logic.
- Wildcard reward imports, path-based model imports, and stringly typed dictionaries make internal
  contracts implicit. Introduce typed configuration and result structures without adding a large
  dependency solely for validation.
- Add fast tests for reset/step invariants, seeded maps and spawns, simultaneous collisions,
  visibility occlusion, component map merge/split behavior, reward accounting, and observation
  space conformance before changing training behavior.
- Add integration smoke tests for PettingZoo compliance, RLlib environment registration, one PPO
  training iteration, checkpoint restore, and deterministic evaluation.

## Change Guidance

- Treat environment semantics, observation schema, reward design, and model architecture as
  separate changes where possible; each can invalidate learned checkpoints.
- When an observation channel or shape changes, update the environment space, RLModule, configs,
  tests, and checkpoint compatibility policy together.
- Use the environment-owned RNG for every stochastic environment or baseline decision.
- Avoid editing generated maps, experiment outputs, checkpoints, or result figures unless the task
  explicitly targets those artifacts.
