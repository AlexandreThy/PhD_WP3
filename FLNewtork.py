import numpy as np
from matplotlib import pyplot as plt
from math import *
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.gridspec as gridspec
from hdf import *

I1 = 0.025
I2 = 0.045
m2 = 1
l1 = 0.3
l2 = 0.33
s2 = 0.16
K = 1 / 0.06
tau = 0.06

# SHOULDER PUIS ELBOW
a1 = I1 + I2 + m2 * l1 * l1
a2 = m2 * l1 * s2
a3 = I2
PLOTWHOLE = False
Bdyn = np.array([[0.05, 0.025], [0.025, 0.05]])

# 50 , 10 for X,Y tuning

NUM_TARG = 8
START_Y = 50
MOVEMENT_LENGTH = 10


def newton(f, Df, epsilon, max_iter, X, Y, x0=np.array([0.8, 1.5])):
    """
    Compute joint angles given a cartesian position X,Y

    Args:
        f: A function that describes the change of coordinate Angular -> Cartesian, giving the error between current estimate and target values .
        df: Analytical derivative of f for newton step.
        epsilon : Acceptable error threshold
        X : X targetted cartesian position
        Y : Y targetted cartesian position

    Returns:
        [theta_s,theta_e] , the joint angles producing the desired (X,Y) cartesian coordinates

    Example:
        >>> calculate_area(f,df,1e-8,1000,0,30)
        [thetas,thetae]
    """
    xn = x0
    for n in range(0, max_iter):
        fxn = f(xn, X, Y)
        if abs(np.max(np.abs(fxn))) < epsilon:
            return xn
        Dfxn = Df(xn)
        if np.max(np.abs(Dfxn)) < epsilon:
            print("Zero derivative. No solution found.")
            return None
        xn = xn - np.linalg.inv(Dfxn) @ fxn
    print("Exceeded maximum iterations. No solution found.")
    return None


def newtonf(var, X, Y):
    u, v = var
    return np.array(
        [
            33 * np.cos(u + v) + 30 * np.cos(u) - X,
            33 * np.sin(u + v) + 30 * np.sin(u) - Y,
        ]
    )


def newtondf(var):
    u, v = var
    return np.array(
        [
            [-33 * np.sin(u + v) - 30 * np.sin(u), -33 * np.sin(u + v)],
            [33 * np.cos(u + v) + 30 * np.cos(u), 33 * np.cos(u + v)],
        ]
    )


class joint2Ddyn:
    def __init__(self, W, Wout, nodes_number):
        self.dt = 0.01
        self.M = 6
        self.Wout = Wout
        self.N = nodes_number
        self.Win = np.zeros((self.N, 1))
        self.W = W
        self.states = np.zeros((self.N + self.M, 1))
        self.SigmaXi = 1e-4
        return

    @staticmethod
    def sysdyn(x, gamma, u, dt, Wout, W):
        """
        Compute one step of the dynamics of the system, composed of a two joint biomechanical model, and a nonlinear network dynamic

        \ddot{\theta} = M^{-1}(Wout gamma-B \dot{\theta} - C)
        \dot{\gamma} = tanh(W gamma) + u

        Args:
            x : x_t biomechanical state at time t [theta_s, theta_e, omega_s, omega_e, angular_acc_s, angular_acc_e]
            gamma : gamma_t vector of the newtork activity at time t
            u : command
            dt : timestep
            Wout : readout matrix
            W : internal network connectivity

        Returns:
            [x_{t+1},gamma_{t+1}]
        """
        M = np.array(
            [[a1 + 2 * a2 * cos(x[1]), a3 + a2 * cos(x[1])], [a3 + a2 * cos(x[1]), a3]]
        )
        C = np.array(
            [
                -x[3] * (2 * x[2] + x[3]) * a2 * np.sin(x[1]),
                x[2] ** 2 * a2 * np.sin(x[1]),
            ]
        )

        x[0:2] += dt * x[2:4]
        x[2:4] += dt * x[4:6]
        x[4:6] = np.linalg.solve(M, (Wout @ gamma - Bdyn @ (x[2:4]) - C))
        gamma += dt * (np.tanh(W @ gamma) + u)
        return np.concatenate((x, gamma))


def compute_control_gains(
    Num_iter, Duration, motor_cost=1e-4, cost_weights=[1e4, 1e4, 1, 1]
):
    """
    Compute the control gains L

    Args:
        Num_iter : the number of simulation steps
        Duration : The Movement Duration
        motor_cost : the motor cost in the cost function r u^2
        cost_weights : q_ij the weight of the states in the cost function : x^TQx
    Returns:
        L : The control gains
    """
    dt = Duration / Num_iter
    Num_Var = 8

    R = np.array([[motor_cost, 0], [0, motor_cost]])
    w1, w2, w3, w4 = cost_weights
    Q = np.array(
        [
            [w1, 0, 0, 0, 0, 0, -w1, 0],
            [0, w2, 0, 0, 0, 0, 0, -w2],
            [0, 0, w3, 0, 0, 0, 0, 0],
            [0, 0, 0, w4, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-w1, 0, 0, 0, 0, 0, w1, 0],
            [0, -w2, 0, 0, 0, 0, 0, w2],
        ]
    )

    A = np.array(
        [
            [1, 0, dt, 0, 0, 0, 0, 0],
            [0, 1, 0, dt, 0, 0, 0, 0],
            [0, 0, 1, 0, dt, 0, 0, 0],
            [0, 0, 0, 1, 0, dt, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    B = np.zeros((8, 2))
    B[4, 0] = dt
    B[5, 1] = dt

    S = Q
    L = np.zeros((Num_iter - 1, 2, Num_Var))

    for k in range(Num_iter - 1):
        L[Num_iter - 2 - k] = np.linalg.inv(R + B.T @ S @ B) @ B.T @ S @ A
        S = A.T @ S @ (A - B @ L[Num_iter - 2 - k])

    return L


def compute_nonlinear_command(L, x, Wout, W, gamma):
    """
    Applies the feedback linearization technique to compute the nonlinear command

    Args:
        L : The control gains
        x : The state of the system at time t
        Wout : readout matrix
        W : internal network connectivity
        gamma : the state of the newtork

    Returns:
        u : the nonlinear command to be send to the network
    """
    M = np.array(
        [[a1 + 2 * a2 * cos(x[1]), a3 + a2 * cos(x[1])], [a3 + a2 * cos(x[1]), a3]]
    )

    Mdot = np.array(
        [
            [-2 * a2 * sin(x[1]) * x[3], -a2 * sin(x[1]) * x[3]],
            [-a2 * sin(x[1]) * x[3], 0],
        ]
    )

    Cdot = np.array(
        [
            -a2 * x[5] * (2 * x[2] + x[3]) * sin(x[1])
            - a2 * x[3] * (2 * x[4] + x[5]) * sin(x[1])
            - a2 * x[3] ** 2 * (2 * x[2] + x[3]) * cos(x[1]),
            2 * x[2] * x[4] * a2 * sin(x[1]) + x[2] ** 2 * a2 * cos(x[1]) * x[3],
        ]
    )

    v = -L @ x
    gammadot = (
        M @ v + Cdot + Bdyn @ np.array([x[4], x[5]]) + Mdot @ np.array([x[4], x[5]])
    )
    u = np.linalg.pinv(Wout) @ gammadot - np.tanh(W @ gamma)
    return u


# now define the 8 condition reaching controller
def eightCondReach(params: dict) -> dict:
    """Simulates an eight-condition reaching task with control gains and neural network dynamics."""
    num_steps = params["num_iter"]
    duration = params["duration"]
    bodyins = joint2Ddyn(params["W"], params["Wout"], params["N"])
    L = compute_control_gains(num_steps, duration)
    num_targconditions = NUM_TARG
    num_states = int(params["N"]) + 6
    dt = duration / num_steps
    all_states = np.zeros((num_steps, num_targconditions, num_states + 2))
    all_readout = np.zeros((num_steps, num_targconditions, 2))

    for i in range(num_targconditions):
        angles = np.linspace(0, 2 * pi, num_targconditions + 1)[:-1]
        st1, st2 = newton(newtonf, newtondf, 1e-8, 1000, 0, START_Y)
        tg1, tg2 = newton(
            newtonf,
            newtondf,
            1e-8,
            1000,
            MOVEMENT_LENGTH * cos(angles[i]),
            START_Y + MOVEMENT_LENGTH * sin(angles[i]),
        )
        all_states[0, i, :] = np.concatenate(
            ([st1, st2, 0, 0, 0, 0, tg1, tg2], np.zeros(params["N"]))
        )

        for j in range(num_steps - 1):
            cur_state = all_states[j, i, :]
            u = compute_nonlinear_command(
                L[j], cur_state[:8], bodyins.Wout, bodyins.W, cur_state[8:]
            )
            next_state = bodyins.sysdyn(
                cur_state[:8], cur_state[8:], u, dt, bodyins.Wout, bodyins.W
            )
            all_states[j + 1, i, :] = next_state
            all_readout[j + 1, i] = bodyins.Wout @ next_state[8:]

    return {"states": all_states, "L": L, "readout": all_readout}


def setParams(W=None, Wout=None) -> dict:
    """Sets parameters for the eight condition reaching task."""
    rtime = 0.6
    num_iter = 60
    return {
        "num_iter": num_iter,
        "W": W,
        "duration": rtime,
        "N": W.shape[0],
        "Wout": Wout,
    }


def runOnce(params: dict) -> tuple:
    """Runs the eight condition reaching task once with given parameters."""
    results = eightCondReach(params)
    behavior = results["states"][:, :, :2]
    network = results["states"][:, :, 8:]
    L = results["L"]
    behavior_gains = L[:, 0, :]
    readout = results["readout"]
    return behavior, network, behavior_gains, readout


def compute_individual_responses(network_activity, n, peak_vel_iter):
    responses = np.zeros((100, n))
    for i in range(100):
        responses[i] = np.sum(network_activity[0, :peak_vel_iter, :, i], axis=0)
    return responses


def get_joint_angles_2():
    theta = np.linspace(0, 2 * pi, NUM_TARG + 1)[:-1]
    vx = MOVEMENT_LENGTH * np.cos(theta)
    vy = MOVEMENT_LENGTH * np.sin(theta)
    V = np.stack((vx, vy), axis=1)  # shape [T, 2]

    return V


def get_joint_angles(N=NUM_TARG):
    joint_angles = np.zeros((N, 2))
    for i, angles in enumerate(np.linspace(0, 2 * pi, N + 1)[:-1]):
        st1, st2 = newton(newtonf, newtondf, 1e-8, 1000, 0, START_Y)
        tg1, tg2 = newton(
            newtonf,
            newtondf,
            1e-8,
            1000,
            MOVEMENT_LENGTH * cos(angles),
            START_Y + MOVEMENT_LENGTH * sin(angles),
        )
        joint_angles[i] = np.array([tg1 - st1, tg2 - st2])
        # joint_angles[i]/= np.linalg.norm(joint_angles[i])
    return joint_angles


if __name__ == "__main__":
    num_runs = 1
    all_behavior, all_network, all_behavior_gains, all_readout = [], [], [], []
    W, Wout = load_networks_by_spectral_radius()
    for i in range(num_runs):
        params = setParams(W, Wout)
        behavior, network, behavior_gains, readout = runOnce(params)
        all_behavior.append(behavior)
        all_network.append(network)
        all_behavior_gains.append(behavior_gains)
        all_readout.append(readout)

    all_behavior = np.array(all_behavior)
    all_network = np.array(all_network)
    all_behavior_gains = np.array(all_behavior_gains)
    all_readout = np.array(all_readout)
    plt.style.use("seaborn-v0_8-darkgrid")  # Nice background style
    colors = plt.cm.viridis(np.linspace(0, 1, NUM_TARG))  # Color map for trajectories

    responses = compute_individual_responses(all_network, NUM_TARG, 30)
    Target_prefered = np.zeros(8)
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")
    for i in range(100):
        model = LinearRegression().fit(get_joint_angles_2(), responses[i].T)
        beta = model.coef_
        preferred_angle = np.arctan2(beta[1], beta[0])
        Target_prefered[int(preferred_angle // (np.pi / 4))] += 0.1
    # Bar plot for preferred direction histogram
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    width = 2 * np.pi / 8 * 0.8  # narrower bars for spacing
    ax.bar(
        angles + width / 2,
        Target_prefered,
        width=width,
        color="mediumseagreen",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.7,
    )

    # Plot target directions as colored dots
    colors = plt.cm.viridis(np.linspace(0, 1, NUM_TARG))
    joint_targets = get_joint_angles(NUM_TARG)

    # Clean up ticks and radial lines
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(True)
    ax.annotate(
        r"$X$",
        xy=(0, 3.5),
        xytext=(5, 5),  # offset in points
        textcoords="offset points",
        fontsize=15,
        color="black",
        ha="left",
        va="bottom",
    )
    ax.annotate(
        r"$Y$",
        xy=(np.pi / 2, 3.5),
        xytext=(5, 5),  # offset in points
        textcoords="offset points",
        fontsize=15,
        color="black",
        ha="left",
        va="bottom",
    )

    plt.title("Preference Direction of neurons\n with joint interaction")
    plt.plot(np.zeros(100), np.linspace(0, 4, 100), color="black")
    plt.plot(np.ones(100) * pi / 2, np.linspace(0, 4, 100), color="black")
    plt.tight_layout()
    plt.savefig("PD_JI.png", dpi=200)
    plt.show()
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")
    # Bar plot for preferred direction histogram
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    width = 2 * np.pi / 8 * 0.8  # narrower bars for spacing
    responses = compute_individual_responses(np.abs(all_network), NUM_TARG, 30)

    ax.bar(
        angles + width / 2,
        np.mean(responses,axis = 0),
        width=width,
        color="red",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.7,
    )

    # Plot target directions as colored dots
    colors = plt.cm.viridis(np.linspace(0, 1, NUM_TARG))
    joint_targets = get_joint_angles(NUM_TARG)

    # Clean up ticks and radial lines
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(True)
    #ax.annotate(
    #    r"$X$",
    #    xy=(0, 3.5),
    #    xytext=(5, 5),  # offset in points
    #    textcoords="offset points",
    #    fontsize=15,
    #    color="black",
    #    ha="left",
    #    va="bottom",
    #)
    #ax.annotate(
    #    r"$Y$",
    #    xy=(np.pi / 2, 3.5),
    #    xytext=(5, 5),  # offset in points
    #    textcoords="offset points",
    #    fontsize=15,
    #    color="black",
    #    ha="left",
    #    va="bottom",
    #)

    plt.title("Activity of neurons\n with joint interaction")
    #plt.plot(np.zeros(100), np.linspace(0, 4, 100), color="black")
    #plt.plot(np.ones(100) * pi / 2, np.linspace(0, 4, 100), color="black")
    plt.tight_layout()
    plt.savefig("PD_JI.png", dpi=200)
    plt.show()
    if PLOTWHOLE:
        fig = plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(4, 3, hspace=0.5)
        ax1 = fig.add_subplot(gs[0:2, :])
        # shape (2,), weights for [theta_s, theta_e]
        for i in range(NUM_TARG):
            thetas = behavior[:, i, 0]
            thetae = behavior[:, i, 1]

            x = 33 * np.cos(thetas + thetae) + 30 * np.cos(thetas)
            y = 33 * np.sin(thetas + thetae) + 30 * np.sin(thetas)

            ax1.plot(x, y, color=colors[i], linewidth=3.5)
            ax1.scatter(
                x[-1], y[-1], color=colors[i], edgecolor="black", zorder=3, s=80
            )  # Start points

        ax1.set_title("Feedback Linearization control \n of nonlinear network")
        ax1.set_aspect("equal", adjustable="box")
        for side in ["left", "right", "bottom", "top"]:
            ax1.spines[side].set_visible(False)
        ax1.set_yticks([])
        ax1.set_xticks([])
        ax1.set_xlabel("")
        ax1.set_ylabel("")
        axes = [
            fig.add_subplot(gs[2, 0]),
            fig.add_subplot(gs[2, 1]),
            fig.add_subplot(gs[2, 2]),
        ]
        for k in range(3):
            ax = axes[k]

            for i in range(NUM_TARG):
                ax.plot(
                    np.linspace(0, 600, 60),
                    all_network[0, :, i, k],
                    label="Target " + str(i),
                    color=colors[i],
                )
            ax.set_title("Neuron " + str(k))
            ax.set_xlabel("Time [ms]")
            if k == 0:
                ax.set_ylabel("Activity")

        ax = fig.add_subplot(gs[3, :])

        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Average Absolute \n Network Activity ")

        for i in range(NUM_TARG):
            ax.plot(
                np.linspace(0, 600, 60),
                np.mean(np.abs(all_network[0, :, i, :]), axis=1),
                color=colors[i],
            )

        plt.savefig("Preferential_Direction2.png", dpi=100)
        plt.show()
