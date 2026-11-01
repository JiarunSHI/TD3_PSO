import numpy as np

def update_particle(p, gbest_pos, action):
    w, c1, c2 = action

    r1, r2 = np.random.rand(), np.random.rand()

    p.vel = (
        w * p.vel
        + c1 * r1 * (p.pbest - p.pos)
        + c2 * r2 * (gbest_pos - p.pos)
    )

    # b机制

    p.vel += np.random.randn(*p.vel.shape)*0.1



    p.pos += p.vel