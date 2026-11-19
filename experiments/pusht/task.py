"""JEPA-TTT PushT geometry, observations and the two paper dynamics shifts."""

import cv2
import numpy as np

if __package__:
    from .paths import add_upstream_paths
else:
    from paths import add_upstream_paths
add_upstream_paths()
from stable_worldmodel.envs.pusht.env import PushT


class PushTTask:
    """The model commands normalized arena targets. Score is T-block area coverage."""

    def __init__(self, shift="original"):
        if shift not in ("original", "action", "contact"):
            raise ValueError(shift)
        self.shift = shift
        self.base = PushT(relative=False, with_target=True, resolution=224)
        self.substep_contact_pairs = 0
        self.moving_contact_substeps = 0
        self.max_contact_rotation_error = 0.0
        self.max_free_rotation = 0.0

    def reset(self, seed):
        base = self.base
        # The upstream reset goal can include a pusher pose; only T coverage sets success.
        base.reset(
            seed=int(seed),
            options={
                "goal_state": np.array(
                    [256.0, 350.0, 256.0, 256.0, np.pi / 4, 0.0, 0.0]
                )
            },
        )
        self.shapes = tuple(base.block.shapes)
        target = base._get_goal_pose_body(base.goal_pose)
        self.vertices = [[v for v in shape.get_vertices()] for shape in self.shapes]
        self.targets = self.polys(target)
        self.target_area = sum(cv2.contourArea(poly) for poly in self.targets)
        if self.shift == "contact":
            space = base.space
            self.previous_contact = False
            self.new_contact = False

            def contact(arbiter, sp, data):
                base._handle_collision(arbiter, sp, data)
                bodies = {shape.body for shape in arbiter.shapes}
                if bodies == {base.agent, base.block}:
                    self.new_contact = True

            space.on_collision(0, 0, post_solve=contact)
            original_step = space.step

            def contact_step(dt):
                self.new_contact = False
                old = base.block.position
                original_step(dt)
                after = base.block.position
                delta = after - old
                # Pymunk integrates the impulse at the following position update. Include
                # the release substep if the immediately preceding solve had agent contact.
                if self.new_contact or self.previous_contact:
                    rotated = (-delta.y, delta.x)
                    base.block.position = old + rotated
                    space.reindex_shapes_for_body(base.block)
                    check = base.block.position - old
                    self.max_contact_rotation_error = max(
                        self.max_contact_rotation_error,
                        abs(check.x + delta.y),
                        abs(check.y - delta.x),
                    )
                    self.substep_contact_pairs += 1
                    self.moving_contact_substeps += int(delta.length > 1.0e-8)
                else:
                    self.max_free_rotation = max(
                        self.max_free_rotation, (base.block.position - after).length
                    )
                self.previous_contact = self.new_contact

            space.step = contact_step
        return self.state()

    def state(self):
        b = self.base
        return np.array(
            [*b.agent.position, *b.block.position, b.block.angle, *b.agent.velocity],
            dtype=np.float64,
        )

    def polys(self, body):
        return [
            np.array([body.local_to_world(v) for v in vertices], dtype=np.float32)
            for vertices in self.vertices
        ]

    def coverage(self):
        # Native T consists of two convex pieces with disjoint interiors.
        cur = self.polys(self.base.block)
        covered = sum(
            float(cv2.intersectConvexConvex(a, b)[0]) for a in cur for b in self.targets
        )
        return float(np.clip(covered / self.target_area, 0.0, 1.0))

    def target(self, command):
        target = (np.clip(command, -1.0, 1.0) + 1.0) * (self.base.window_size / 2)
        if self.shift == "action":
            p = np.asarray(self.base.agent.position)
            dx, dy = target - p
            target = p + np.array([-dy, dx])
        return target

    def step(self, command):
        self.base.step(self.target(command))
        value = self.coverage()
        return value, value >= self.base.success_threshold

    def render(self):
        return self.base.render()

    def close(self):
        self.base.close()

    def sample_command(self, rng):
        normalize = self.base.window_size / 2
        if rng.random() < 0.5:
            target = np.asarray(self.base.block.position) + rng.normal(
                0.0, 60.0, size=2
            )
        else:
            target = rng.uniform(0.0, self.base.window_size, size=2)
        return np.clip(target / normalize - 1.0, -1.0, 1.0).astype(np.float32)
