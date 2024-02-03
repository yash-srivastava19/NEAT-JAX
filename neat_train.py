import os
import shutil
import jax

from hyp import hyp
from NEAT import NEATJAX  # Custom NEAT JAX Solver.

from dataclasses import dataclass

from evojax import Trainer
from evojax import util
from evojax.task.slimevolley import SlimeVolley
from evojax.policy.mlp import MLPPolicy

@dataclass
class TrainingParams:
    NUM_TESTS = 100
    N_REPEATS = 16
    MAX_ITER = 50
    TEST_INTERVAL = 50
    LOG_INTERVAL = 10
    MAX_STEPS = 3000
    LOG_DIR = "./log/slimevolley"

tp = TrainingParams()

@dataclass
class PolicyParams:
    HIDDEN_SIZE = 20
    OUT_ACT_FN = 'tanh'

pp = PolicyParams()

# Logging to make it easier to debug.
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

logger = util.create_logger(
    name='SlimeVolley', log_dir=tp.LOG_DIR, debug=True)

logger.info('EvoJAX SlimeVolley')

logger.info('=' * 30)

# Define training tasks and policy(we'll go with MLP Policy as of now).
train_task = SlimeVolley(test=False, max_steps=tp.MAX_STEPS)
test_task = SlimeVolley(test=True, max_steps=tp.MAX_STEPS)

policy = MLPPolicy(
    input_dim=train_task.obs_shape[0],
    hidden_dims=[pp.HIDDEN_SIZE, ],
    output_dim=train_task.act_shape[0],
    output_act_fn=pp.OUT_ACT_FN,
)

# Here, we'll have our custom NEAT algo for solving this.
solver = NEATJAX(hyp)

# Define the trainer and then start the training.
trainer = Trainer(
    policy=policy,
    solver=solver,
    train_task=train_task,
    test_task=test_task,
    max_iter=MAX_ITER,
    log_interval=LOG_INTERVAL,
    test_interval=TEST_INTERVAL,
    n_repeats=N_REPEATS,
    n_evaluations=NUM_TESTS,
    seed=SEED,
    log_dir=LOG_DIR,
    logger=logger,
)
trainer.run(demo_mode=False)

# See the best model statistics(and run in demo mode).
src_file = os.path.join(LOG_DIR, 'best.npz')
tar_file = os.path.join(LOG_DIR, 'model.npz')

shutil.copy(src_file, tar_file)

trainer.model_dir = LOG_DIR
trainer.run(demo_mode=True)

# Now, we'll go the part of making the GIF of the end results.

#First, we'll get the task and policy reset functions and get the best parameters from the trainer run.  
task_reset_fn = jax.jit(test_task.reset)
policy_reset_fn = jax.jit(policy.reset)
step_fn = jax.jit(test_task.step)
action_fn = jax.jit(policy.get_actions)
best_params = trainer.solver.best_params[None, :]
key = jax.random.PRNGKey(0)[None, :]

# The loop to create a GIF of slimevolley play.
task_state = task_reset_fn(key)
policy_state = policy_reset_fn(task_state)
screens = []
for _ in range(MAX_STEPS):
    action, policy_state = action_fn(task_state, best_params, policy_state)
    task_state, reward, done = step_fn(task_state, action)
    screens.append(SlimeVolley.render(task_state))

gif_file = os.path.join(LOG_DIR, 'slimevolley.gif')
screens[0].save(gif_file, save_all=True, append_images=screens[1:],
                duration=10, loop=0)

logger.info('GIF saved to {}.'.format(gif_file))

