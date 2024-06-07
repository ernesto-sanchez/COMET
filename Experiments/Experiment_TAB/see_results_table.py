import pickle
import matplotlib.pyplot as plt
import os

script_path = os.path.dirname(__file__)
with open('results.pkl', 'rb') as f:
    results = pickle.load(f)

x_values = [0,0.1, 0.2,0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1]  # Define the values for the x-axis
colors = ['red', 'blue', 'green', 'orange']  # Define the colors for each line

for model, color in zip(results.keys(), colors):
    y_values = [results[model][tab].loc['counterfactual','Test'] for tab in x_values]  # Get the error values for each model
    plt.plot(x_values, y_values, color=color, label=model)  # Plot the line for each model

plt.xlabel('TAB')  # Set the label for the x-axis
plt.ylabel('CATE')  # Set the label for the y-axis
plt.legend()  # Show the legend for the lines
plt.savefig(os.path.join(script_path, 'results.pdf'), dpi=300)  # Save the plot as a PDF file with high resolution
plt.show()  # Display the plot