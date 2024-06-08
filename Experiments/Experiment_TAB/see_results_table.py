import pickle
import matplotlib.pyplot as plt
import os


with open('/cluster/work/medinfmk/STCS_swiss_transplant/AI_Organ_Transplant_Matching/code/code_ernesto/comet_cluster/Experiments/Experiment_TAB/results.pkl', 'rb') as f:
    results = pickle.load(f)
    
script_path = os.path.dirname(__file__)
x_values = [0,0.1, 0.2,0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1]  # Define the values for the x-axis
# colors = ['red', 'blue', 'green', 'orange']  # Define the colors for each line
colors = ['red', 'green']  # Define the colors for each line


# ['S_Learner()', 'DoubleML()', 'DRLearner()', 'T_Learner()']

for model, color in zip(['S_Learner()', 'T_Learner()'], colors):
    y_values = [results[model][tab].loc['counterfactual','Train'] for tab in x_values]  # Get the error values for each model
    plt.plot(x_values, y_values, color=color, label=model)  # Plot the line for each model

plt.xlabel('TAB')  # Set the label for the x-axis
plt.ylabel('PEHE')  # Set the label for the y-axis
plt.legend()  # Show the legend for the lines
plt.tight_layout()  # Adjust the layout
plt.savefig(os.path.join(script_path, 'results.pdf'), dpi=300)  # Save the plot as a PDF file with high resolution
plt.show()
