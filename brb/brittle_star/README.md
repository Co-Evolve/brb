# Light fleeing brittle stars

## Description

Optimize a controller to steer the brittle star away from light.

## Action Space

Dimensionality and meaning depend on the morphological configuration. The robot can use tendon-based actuation (
currently deprecated) or p-controller-based actuation.

### Tendons

Currently deprecated.

### P-Controllers

Every segment has two p-controllers: one for the in-plane joint angle, and one for the out-of-plane joint angle.
The action boundaries are equal to the joint limits defined in the morphology's specification.

The total number of actions is thus equal to $number\\_of\\_arms \times number\\_of\\_segments\\_per\\_arm \times 2$.

## Observation Space

| Observation                          | Meaning                                                                                                                                                                   | Min  | Max | Shape                                                                   |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|-----|-------------------------------------------------------------------------|
| task/time                            | The current simulation time.                                                                                                                                              | 0    | Inf | 1                                                                       |
| task/delta_time                      | Amount of time by which the simulation has progressed in this step.                                                                                                       | 0    | Inf | 1                                                                       |
| task/light_per_segment               | The normalised light income per segment.                                                                                                                                  | 0    | 1   | $number\\_of\\_arms \times number\\_of\\_segments\\_per\\_arm$          |
| task/distance_travelled_along_x_axis | The total distance travelled along the x-axis, relative to the starting position.                                                                                         | -Inf | Inf | 1                                                                       |
| morphology/angular_velocity          | The angular velocity of the disc.                                                                                                                                         | -5   | 5   | 3                                                                       |
| morphology/linear_velocity           | The linear velocity of the disc.                                                                                                                                          | -5   | 5   | 3                                                                       |
| morphology/touch_per_tendon_plate    | Discretized contact flags per segment plate. Each segment has two plates that register contact forces. This value will be 1 if a contact was registered and -1 otherwise. | -1   | 1   | $number\\_of\\_arms \times number\\_of\\_segments\\_per\\_arm \times 2$ |

## Rewards

The reward is defined as the difference in the total amount of light income between the previous simulation step and the
current simulation step.
The reward will thus be positive when the total amount of light income has reduced.

