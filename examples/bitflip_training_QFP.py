import torch
import math
import numpy as np

from protocolopt.potentials import GeneralCoupledPotential
from protocolopt.potentials.quartic import QFP_Potential
# from protocolopt.types import StateSpace, ControlVector
from protocolopt.protocols import LinearPiecewise
from protocolopt.simulators import EulerMaruyama
from protocolopt.losses import StandardLogicGateLoss
from protocolopt import ProtocolOptimizer
from protocolopt.sampling import ConditionalFlow, LaplaceApproximation
from protocolopt.callbacks import TrajectoryPlotCallback, ConfusionMatrixCallback, PotentialLandscapePlotCallback, ProtocolPlotCallback
from protocolopt.callbacks.plotting import SaveTrajectoryCallback

try:
    from protocolopt.callbacks import AimCallback
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    print("Aim not available - skipping experiment tracking")

# Device configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")


k_B, h_bar, PHI_0 = 1.380649e-23, 1.054571817e-34, 2.067833831e-15

R, L, C = 100, 5e-12, 1e-12  # ohm, H, F
T = 4.2  # K
k_BT = k_B * T

# constant factor
t_duration = 6

t_c = t_duration * np.sqrt(L * C)
x_c = PHI_0 / (2 * np.pi)  # Convert to appropriate units
v_c = x_c / t_c
m_c = C
nu_c = 1/R

# non-dimensional parameters
nu_1, nu_2 = 2, 1/2
m_1 = 1


# U_0 = float(m_c * x_c**2 / t_c**2 / k_BT) # in units of kBT
U_0 = float(x_c**2 / L / k_BT) # in units of kBT
xi = U_0 * k_BT * t_c**2 / m_c / x_c**2
# U_0 = 1.0



# ProtocolOptimizer parameters
# time_steps = 1000
time_steps = int(t_duration / (1/100))
dt = 1/time_steps
# gamma = 0.1
gamma = (nu_c * t_c / m_c) * nu_1
epsilon = float(np.sqrt(2 * gamma * k_BT / (m_c * v_c**2)))
# gamma = 10 

beta = U_0
mass = m_1
# Protocol parameters
num_coefficients = 16
a_endpoints = [0.0, 0.0]
b_endpoints = [0, 0]

# Training parameters
samples_per_well = 2000 
training_iterations = 400
learning_rate = 0.25
alpha = 2.0  # endpoint_weight
alpha_1 = 0.1  # var_weight
alpha_2 = 10  # work_weight
alpha_3 = 5e-4  # smoothness_weight
alpha_4 = 0. # out_of_eqm_weight

# Additional parameters
spatial_dimensions = 1
mcmc_warmup_ratio = 0.1
mcmc_starting_spatial_bounds = torch.tensor([[-4.0, 4.0]], device=device)

# Calculate centers
centers = 2.04

# Create endpoints tensor
endpoints = torch.tensor([
    [a_endpoints[0], a_endpoints[1]], 
    [b_endpoints[0], b_endpoints[1]]
], device=device)

# Create initial coefficient guess
initial_coeff_guess = 0.1*torch.randn((2, num_coefficients), device=device)



# Instantiate Protocol (LinearPiecewise)
protocol = LinearPiecewise(
    control_dim=2,
    time_steps=time_steps,
    knot_count=num_coefficients+2,
    initial_coeff_guess=initial_coeff_guess,
    endpoints=endpoints
)

# Instantiate Potential (GeneralCoupledPotential)
potential = QFP_Potential(compile_mode=True)

# Instantiate Simulator (EulerMaruyama)
simulator = EulerMaruyama(
    mode='underdamped',
    gamma=gamma,
    beta=beta,
    xi=xi,
    mass=mass,
    dt=dt,
    compile_mode=True
)

# Create loss function (StandardLogicGateLoss)
midpoints = torch.tensor([0.0], device=device)
bit_locations = torch.tensor([[-centers], [centers]], device=device)
truth_table = {0: ['1'], 1: ['0']}

loss = StandardLogicGateLoss(
    midpoints=midpoints,
    truth_table=truth_table,
    bit_locations=bit_locations,
    endpoint_weight=alpha,
    work_weight=alpha_2,
    var_weight=alpha_1,
    smoothness_weight=alpha_3,
    out_of_eqm_weight=alpha_4,
    exponent=2
)


# Create callbacks
callbacks = []

# Add plotting callbacks
trajectory_callback = TrajectoryPlotCallback(
    save_dir='figs',
    plot_frequency=None,
    num_trajectories=100
)
callbacks.append(trajectory_callback)

confusion_matrix_callback = ConfusionMatrixCallback(
    save_dir='figs',
    plot_frequency=None
)
callbacks.append(confusion_matrix_callback)

potential_landscape_callback = PotentialLandscapePlotCallback(
    save_dir='figs',
    plot_frequency=None
)
callbacks.append(potential_landscape_callback)

coefficient_callback = ProtocolPlotCallback(
    save_dir='figs',
    plot_frequency=None
)

callbacks.append(coefficient_callback)


save_trajectory_callback = SaveTrajectoryCallback()
callbacks.append(save_trajectory_callback)

# IMPORTANT: AimCallback must be last
# Add Aim callback if available
if AIM_AVAILABLE:
    aim_callback = AimCallback(
        experiment_name='bitflip_training',
        log_system_params=False
    )
    callbacks.append(aim_callback)


# init_cond_generator = ConditionalFlow(
#     dt=dt,
#     gamma=gamma,
#     mass=mass,
#     device=device,
#     spatial_dimensions=spatial_dimensions,
#     time_steps=time_steps,
#     beta=beta,
#     starting_bounds=mcmc_starting_spatial_bounds,
#     samples_per_well=samples_per_well,
#     chains_per_well=1,
#     warmup_ratio=mcmc_warmup_ratio
# )

num_samples = samples_per_well * (2**spatial_dimensions)

init_cond_generator = LaplaceApproximation(
    dt=dt,
    gamma=gamma,
    mass=mass,
    centers=bit_locations,
    device=device,
    beta=beta,
    spatial_dimensions=spatial_dimensions,
    time_steps=time_steps,
    energy_factor=U_0,
    num_samples=num_samples
)

# Instantiate ProtocolOptimizer
simulation = ProtocolOptimizer(
    potential=potential,
    simulator=simulator,
    loss=loss,
    protocol=protocol,
    initial_condition_generator=init_cond_generator,
    epochs=training_iterations,
    learning_rate=learning_rate,
    callbacks=callbacks,
    scheduler_kwargs={'T_0': training_iterations // 5},
    scheduler_restart_decay=0.75
)

if __name__ == '__main__':
    print("Starting bitflip training...")

    # initial_pos, initial_vel, noise = init_cond_generator.generate_initial_conditions(potential=potential, protocol=protocol, loss=loss)
    simulation.train()

    print("Training complete!")
