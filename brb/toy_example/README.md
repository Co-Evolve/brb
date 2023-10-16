# Toy Example

## Description

Toy example for the implementation of a morphology and an environment. The morphology is similar
to [gymnasium's Ant](https://www.gymlibrary.dev/environments/mujoco/ant/), although here we also parameterize the number
of
arms and the number of segments per arm. The objective is to move the ant towards a target (red sphere).

## Code structure

The code structure presented here should be mimicked for new morphologies and environments.

- [name of animal]/
    - morphology/
        - morphology.py (implements the instantiation of a given morphology specification; i.e. maps the parameterised
          morphology specification to the XML used by MuJoCo. This uses the MuJoCo wrappers, implemented
          in [mujoco-utils](https://github.com/Co-Evolve/mujoco-utils), of the base classes defined
          in [FPRS](https://github.com/Co-Evolve/fprs.)
          using [dm_control.mjcf](https://github.com/google-deepmind/dm_control/blob/main/dm_control/mjcf/README.md))
        - observables.py (implements the morphology specific observations; i.e. observations that are acquired by
          sensors)
        - parts/ (contains the different parts that make up the morphology)
        - specification/
            - specification.py (implements the parameterised morphology specification
              using [FPRS](https://github.com/Co-Evolve/fprs). Initially, all parameters are FixedParamers TODO)
            - default.py (defines default parameter values for the specification)
    - arena/
        - assets/ (optional assets directory)
        - [descriptive name of arena].py (implements the environment)
    - task/
        - [descriptive name of task].py (handles the interaction between the morphology and the arena and defines the
          objective and objective related observations)

## Example scripts

-`export_mjcf.py`: shows how to export a morphology's xml format, to be viewed
in [MuJoCo's native viewer](https://github.com/google-deepmind/mujoco/releases).

-`dm_control_viewer.py`: shows how to instantiate a morphology and load it into an environment. The running environment
is then visualised
using [dm_control's viewer](https://github.com/google-deepmind/dm_control/blob/main/dm_control/viewer/README.md).

-`gym_interface.py`: shows how to wrap the dm_control environment with
a [gymnasium interface](https://gymnasium.farama.org/) (as more often used for optimization). The running environment is
then visualised using opencv.

-`parameters.py`: shows how to adapt the specification in order to make specific parameters mutable.  