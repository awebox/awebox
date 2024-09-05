import matplotlib.pyplot as plt
import numpy as np
import math

# Calculate normalized lift power
def liftPower(e, sF, aerodynamic) :
    return (aerodynamic/(1+(1/4)*aerodynamic*sF)**2)*e*((1-e)**2)

# Calculate lift mode induction factor
def liftModeInduc(e, sF, aerodynamic) :
    return (1/4)*sF*aerodynamic*e/((1+(1/4)*sF*aerodynamic)*e)

def main():
    # Declare variables
    a = 0.5
    CL = 1.0
    CD = 0.1
    solidityFactor = 0.001
    aerodynamic = CL*(math.pow(CL/CD, 2))

    # Create two plots
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle("Lift Power and Induction Factor")

    # Set the interval für the Reel-out speed
    e = np.arange(0.0, 1.0, 0.01)
    #e1 = np.arange(0.0, 1.0, 0.1)

    # Calculate normalized lift power plots, varying the solidity Factor 
    sF = solidityFactor
    while sF < 0.011 :
        ax1.plot(e, liftPower(e, sF, aerodynamic), "k", 1/3, liftPower(1/3, sF, aerodynamic), "bo")
        sF += 0.001

    sF = 0.0345
    ax1.plot(e, liftPower(e, sF, aerodynamic), "k", 1/3, liftPower(1/3, sF, aerodynamic), "bo")

    # Calculate lift mode induction factor plots, varying the solidity Factor
    sF = solidityFactor
    while sF < 0.011 :
        ax2.plot(e, liftModeInduc(e, sF, aerodynamic), "k", 1/3, liftModeInduc(1/3, sF, aerodynamic), "bo")
        sF += 0.001

    sF = 0.0345
    ax2.plot(e, liftModeInduc(e, sF, aerodynamic), "k", 1/3, liftModeInduc(1/3, sF, aerodynamic), "bo")

    # Set the axis
    ax1.set_xlabel("Reel-out speed")
    ax1.set_ylabel("Normalized Lift Power")
    ax1.axis([0, 1, 0, 16])
    ax2.set_xlabel("Reel-out speed")
    ax2.set_ylabel("Lift mode Induction Factor")
    ax2.axis([0, 1, 0, 0.6])

    # Adjust the subplot layout
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, hspace=0.5)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()