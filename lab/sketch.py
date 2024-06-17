import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import streamlit as st


def dot_cluster():
    # Number of points
    num_points = 100

    # Generate random x-coordinates
    x = np.random.rand(num_points)

    # Define the angle in degrees and convert to radians
    angle_degrees = 20
    angle_radians = np.radians(angle_degrees)

    # Calculate the slope of the line corresponding to the angle
    slope = np.tan(angle_radians)

    # Calculate y-coordinates with some noise added
    noise = np.random.normal(scale=0.05, size=num_points)  # Adjust the scale for more or less noise
    y = slope * x + noise + 0.3

    with plt.xkcd():
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_ylim([0, 1])
        ax.set_xticks([])
        ax.set_yticks([])

        reg = linear_model.LinearRegression()
        reg.fit(x.reshape(-1, 1), y)
        x_line = np.linspace(0, 1, 100)
        y_line = (reg.coef_[0] * x_line) + reg.intercept_
        ax.plot(x_line, y_line, 'r--', label='1st Principal Component')  # 'r--' for red dotted line
        normal_slope = -1 / slope
        mid_x = 0.5
        mid_y = slope * mid_x
        x_normal_line = np.linspace(mid_x - 0.5, mid_x + 0.5, 100)
        y_normal_line = normal_slope * (x_normal_line - mid_x) + mid_y
        ax.plot(x_normal_line, y_normal_line, 'g--', label='2nd Principal Component')  # 'r--' for red dotted line
        plt.legend()
        print(slope)

        st.pyplot(fig)


def vectors():
    with plt.xkcd():
        fig, ax = plt.subplots()
        ax.arrow(0, 0, 0.5, 0.3, head_width=0.01)
        ax.scatter(x=[0.525], y=[0.315])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim([-0.03, 1])
        ax.set_ylabel("Royalty")
        ax.set_xlabel("Gender")
        ax.set_title("'Queen' in Embedding Space (Bad Example)")

        st.pyplot(fig)

def main():
    # dot_cluster()
    vectors()