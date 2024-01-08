import jax
import jax.numpy as jnp

from brb.toy_example.arena.arena import MJCFPlaneWithTargetArena, PlaneWithTargetArenaConfiguration
from brb.toy_example.mjc.env import ToyExampleEnvironmentConfiguration
from brb.toy_example.mjc.example_usage_single import post_render
from brb.toy_example.mjx.env import ToyExampleMJXEnvironment
from brb.toy_example.morphology.morphology import MJCFToyExampleMorphology
from brb.toy_example.morphology.specification.default import default_toy_example_morphology_specification


def create_mjx_environment(
        environment_configuration: ToyExampleEnvironmentConfiguration
        ) -> ToyExampleMJXEnvironment:
    morphology_specification = default_toy_example_morphology_specification(num_arms=4, num_segments_per_arm=2)
    morphology = MJCFToyExampleMorphology(specification=morphology_specification)
    arena_configuration = PlaneWithTargetArenaConfiguration()
    arena = MJCFPlaneWithTargetArena(configuration=arena_configuration)
    env = ToyExampleMJXEnvironment(
            morphology=morphology, arena=arena, configuration=environment_configuration
            )
    return env


if __name__ == '__main__':
    environment_configuration = ToyExampleEnvironmentConfiguration(
            render_mode="human", camera_ids=[0, 1]
            )
    env = create_mjx_environment(environment_configuration=environment_configuration)

    rng = jax.random.PRNGKey(0)
    rng, subrng = jax.random.split(key=rng, num=2)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    state = jit_reset(rng=subrng)

    done = False
    steps = 0
    fps = 60
    while not done:
        t = state.info["time"]
        action = jnp.ones(env.action_space.shape)
        action = action.at[jnp.arange(0, len(action), 2)].set(jnp.cos(5 * t))
        action = action.at[jnp.arange(1, len(action), 2)].set(jnp.sin(5 * t))
        action = action.at[jnp.arange(len(action) // 2, len(action), 2)].set(
                action[jnp.arange(len(action) // 2, len(action), 2)] * -1
                )

        state = jit_step(state=state, action=action)
        done = bool(state.terminated or state.truncated)
        if steps % int((1 / fps) / environment_configuration.control_timestep) == 0:
            post_render(env.render(state), environment_configuration=environment_configuration)
            print(state.observations["segment_ground_contact"])

        steps += 1
        if done:
            rng, subrng = jax.random.split(key=rng, num=2)
            state = jit_reset(rng=subrng)
            post_render(env.render(state), environment_configuration=environment_configuration)
            done = False
            steps = 0
    env.close()
