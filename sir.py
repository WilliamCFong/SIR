import click
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


def dSIRdt(SIR, t, population, infection_rate, recovery_rate):
    S, I, R = SIR  # Unpack input vector
    dSdt = -infection_rate * S * I / population
    dIdt = -dSdt - recovery_rate * I
    dRdt = recovery_rate * I

    return dSdt, dIdt, dRdt


@click.command()
@click.argument('population', type=int)
@click.option('--beta', '-b', type=float, default=0.07, help='Infection rate /day')
@click.option('--gamma', '-g', type=float, default=0.02, help='Recovery rate /day')
@click.option('--infected_0', '-i', type=int, default=1, help='Number of infected at day 0')
@click.option('--recovered_0', '-r', type=int, default=0, help='Number of \'recovered\' at day 0')
@click.option('--timespan', '-t', type=int, default=31, help='Number of days to sweep')
def sir(population, beta, gamma, infected_0, recovered_0, timespan):

    susceptible_0 = population - infected_0 - recovered_0
    initial_conditions = (susceptible_0, infected_0, recovered_0)

    timespan = np.linspace(0, timespan, num=timespan)

    SIR = odeint(dSIRdt, initial_conditions, timespan, args=(population, beta, gamma))
    S, I, R = SIR.T  # Transpose column wise and unpack

    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(timespan, S, 'b', alpha=0.5, lw=2, label='S')
    ax.plot(timespan, I, 'r', alpha=0.5, lw=2, label='I')
    ax.plot(timespan, R, 'g', alpha=0.5, lw=2, label='R')
    ax.set_xlabel('Days')
    ax.set_ylabel('N People')
    ax.set_ylim(0, population * 1.1)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    plt.show()


if __name__ == '__main__':
    sir()
