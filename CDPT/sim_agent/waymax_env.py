# # This class is a wrapper of Waymax to simulate the environment from WOMD


import numpy as np
from jax import jit
from jax import numpy as jnp
import jax  

from waymax import config as _config
from waymax import datatypes  
from waymax import dynamics
from waymax import env as waymax_env
from waymax import agents
from waymax.agents import actor_core
from waymax.agents import waypoint_following_agent

from typing import List


class WaymaxEnvironment(waymax_env.BaseEnvironment):
    def __init__(
            self,
            dynamics_model: dynamics.DynamicsModel,
            config: _config.EnvironmentConfig,
            log_replay=False
        ):
        """
        Initializes a new instance of the WaymaxEnv class.

        Args:
            dynamics_model (dynamics.DynamicsModel): The dynamics model used for simulating the environment.
            config (_config.EnvironmentConfig): The configuration object for the environment.
            log_replay (bool): If True, use logged data for all valid objects; if False, use IDM for valid objects not controlled by VBD.
        """
        super().__init__(dynamics_model, config)
        
        # Store dynamics model and config
        self._dynamics_model = dynamics_model
        self.config = config
        self.log_replay = log_replay  # Store for use in step_sim_agent

        if log_replay:
            # Expert actor replays logged data for all valid objects
            # Note: is_sdc may be used to identify the SDC, but is_valid controls all objects here
            self.nc_actor = agents.create_expert_actor( 
                dynamics_model=dynamics_model,
                is_controlled_func=lambda state: state.object_metadata.is_valid)
        else:
            # IDM policy for valid objects not controlled by VBD
            # Note: is_sdc is not explicitly used; VBD may control SDC based on test.py logic
            def idm_controlled_func(state: datatypes.SimulatorState) -> jax.Array:  # Fix: use `jax.Array` instead of `jnp.Array`
                # Control valid objects not already controlled by VBD
                # is_controlled tracks VBD-controlled objects; is_valid ensures only valid objects are controlled
                return state.object_metadata.is_valid & ~state.object_metadata.is_controlled
            
            self.nc_actor = waypoint_following_agent.IDMRoutePolicy(
                is_controlled_func=idm_controlled_func,
                desired_vel=30.0,  # Default IDM parameters
                min_spacing=2.0,
                safe_time_headway=2.0,
                max_accel=2.0,
                max_decel=4.0,
                delta=4.0,
                max_lookahead=10,
                lookahead_from_current_position=True,
                additional_lookahead_points=10,
                additional_lookahead_distance=10.0,
                invalidate_on_end=False
            )
    
        # Useful jited functions 
        self.jit_step = jit(self.step)
        self.jit_nc_action = jit(self.nc_actor.select_action)
        self.jit_reset = jit(super().reset)
        
    def step_sim_agent(
        self,
        current_state: datatypes.SimulatorState,
        sim_agent_action_list: List[datatypes.Action]
    ) -> datatypes.SimulatorState:
        """
        Steps the simulation agent.

        Notes:
            - is_controlled: Set to True for VBD-controlled objects; IDM controls remaining valid objects when log_replay=False.
            - is_modeled: Set to True for all controlled or valid objects when log_replay=False to indicate simulation.
            - is_sdc: Not modified; may be used by VBD to identify SDC (set in test.py).
        """
        # Step the non-controlled policy (expert or IDM)
        nc_action_full: actor_core.WaymaxActorOutput = self.jit_nc_action({}, current_state, None, None)

        # Validation check to ensure no object is controlled by multiple policies
        is_controlled_stack = jnp.vstack([action.is_controlled for action in sim_agent_action_list])
        num_controlled = jnp.sum(is_controlled_stack, axis=0)  # (num_agent, 1)
        if jnp.any(num_controlled > 1):
            raise Exception("An agent is controlled by more than one policy")

        # Set the is_controlled flag for nc_actor (IDM or expert)
        # Only apply nc_actor to objects not controlled by VBD
        simple_control = num_controlled == 0

        nc_action = actor_core.WaymaxActorOutput(
            action=nc_action_full.action,
            actor_state=None,
            is_controlled=simple_control & nc_action_full.is_controlled
        )
        
        # Merge actions from VBD and nc_actor
        sim_agent_action_list.append(nc_action)
        action_merged = agents.merge_actions(sim_agent_action_list)

        # Step the environment
        next_state = self.jit_step(current_state, action_merged)
        
        # Update metadata
        next_state.object_metadata.is_controlled = num_controlled > 0
        if not self.log_replay:
            # Mark all controlled or valid objects as modeled when using IDM
            next_state.object_metadata.is_modeled = (
                next_state.object_metadata.is_controlled | next_state.object_metadata.is_valid
            )

        return next_state
        
    def reset(self, state: datatypes.SimulatorState) -> datatypes.SimulatorState:
        """Initializes the simulation state."""
        return self.jit_reset(state)  
    
