#!/bin/bash

# default args are for the online GCRL setting, so we need to change some of them
# for offline d4rl.

args=(
    env.kind=d4rl
    num_workers=12
    # encoder
    agent.quasimetric_critic.model.encoder.arch="[1024,1024,1024]"
    # quasimetric model
    agent.quasimetric_critic.model.quasimetric_model.projector_arch="[1024,1024]"
    # dynamics
    agent.quasimetric_critic.model.latent_dynamics.arch="[1024,1024,1024]"
    agent.quasimetric_critic.losses.latent_dynamics.weight=1
    # critic lr
    agent.quasimetric_critic.losses.critic_optim.lr=5e-4
    agent.quasimetric_critic.losses.critic_optim.cosine_lr_decay_final_mul=0
    # actor
    agent.actor.model.arch="[1024,1024,1024,1024]"
    agent.actor.losses.actor_optim.lr=3e-5
    agent.actor.losses.actor_optim.cosine_lr_decay_final_mul=0
    agent.actor.losses.min_dist.adaptive_entropy_regularizer=False
    agent.actor.losses.min_dist.add_goal_as_future_state=False
    agent.actor.losses.behavior_cloning.weight=0.05
)

exec python -m offline.main "${args[@]}" "${@}"

