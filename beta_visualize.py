import matplotlib.pyplot as plt
import numpy as np




# Define the perturbation levels (including clean) to plot in a specific order
# labels = ["clean", "perturb_1.0", "perturb_2.0", "perturb_3.0", "perturb_4.0", "perturb_5.0"]
num_epochs = 101
epochs = list(range(1, num_epochs + 1))  # List of epoch numbers

beta_values = dict()

beta_values["clean"] = [np.loadtxt("./betas/CIDDS_Combined_clean_.log").tolist()]
beta_values["perturb_1.0"] = [np.loadtxt("./betas/CIDDS_Combined_random_1.log").tolist()]
beta_values["perturb_2.0"] = [np.loadtxt("./betas/CIDDS_Combined_random_2.log").tolist()]
beta_values["perturb_3.0"] = [np.loadtxt("./betas/CIDDS_Combined_random_3.log").tolist()]
# beta_values["perturb_4.0"] = [np.loadtxt("./betas/squirrel_CombinedModel_GCN_nettack_4.0.log").tolist()]
# beta_values["perturb_5.0"] = [np.loadtxt("./betas/squirrel_CombinedModel_GCN_nettack_5.0.log").tolist()]

# beta_values["clean"] = [np.loadtxt("./betas/squirrel_CombinedModel_GCN_clean_meta_.log").tolist()]
# beta_values["perturb_0.1"] = [np.loadtxt("./betas/squirrel_CombinedModel_GCN_meta_0.1.log").tolist()]
# beta_values["perturb_0.2"] = [np.loadtxt("./betas/squirrel_CombinedModel_GCN_meta_0.2.log").tolist()]

labels = beta_values.keys()

# Set up the plot
plt.figure(figsize=(10, 6))

# Plot each version's beta values over epochs
for label in labels:
    plt.plot(epochs, beta_values[label][0][:num_epochs], label=label)

# Add plot details
plt.xlabel("Epoch")
plt.ylabel("Beta Value")
plt.title("Beta Value Across Different Perturbation Levels")
plt.legend(title="Data Version")
plt.grid(True)
plt.tight_layout()


# Save fig
# plt.savefig("beta_values_plot_squirrel_meta_GCN.png", dpi=300, bbox_inches='tight')

plt.savefig("beta_values_plot_CIDDS_random_Combined.png", dpi=300, bbox_inches='tight')

"""

# Create a 3x2 grid of subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))
# fig.suptitle('Trajectory of Beta Values for the Pubmed Dataset', fontsize=16)

# List of models for labeling the rows
models = ['GCN', 'GPRGNN', 'GAT']
num_epochs = [800, 600, 200]

# Define lists to store lines and labels for both Nettack and Meta legends
lines_nettack, labels_nettack = [], []
lines_meta, labels_meta = [], []

# Loop through each subplot and customize
for i in range(3):  # For each row (model)
    for j in range(2):  # For each column (attack method)
        ax = axes[i, j]  # Get the specific subplot

        # Title each column
        if i == 0:  # Top row only
            ax.set_title('Nettack' if j == 0 else 'Meta', fontstyle='italic')

        # Label each row on the left
        if j == 0:
            ax.set_ylabel(f'{models[i]} Results', fontstyle='italic')
            attack_type = "nettack"
        else:
            attack_type = "meta"

        # Load beta values based on the attack type
        beta_values = {}
        if attack_type == "nettack":
            beta_values["clean"] = [
                np.loadtxt(f"./betas/pubmed_CombinedModel_{models[i]}_clean_{attack_type}_.log").tolist()]
            beta_values["perturb_1.0"] = [
                np.loadtxt(f"./betas/pubmed_CombinedModel_{models[i]}_{attack_type}_1.0.log").tolist()]
            beta_values["perturb_2.0"] = [
                np.loadtxt(f"./betas/pubmed_CombinedModel_{models[i]}_{attack_type}_2.0.log").tolist()]
            beta_values["perturb_3.0"] = [
                np.loadtxt(f"./betas/pubmed_CombinedModel_{models[i]}_{attack_type}_3.0.log").tolist()]
            beta_values["perturb_4.0"] = [
                np.loadtxt(f"./betas/pubmed_CombinedModel_{models[i]}_{attack_type}_4.0.log").tolist()]
            beta_values["perturb_5.0"] = [
                np.loadtxt(f"./betas/pubmed_CombinedModel_{models[i]}_{attack_type}_5.0.log").tolist()]
        else:
            beta_values["clean"] = [
                np.loadtxt(f"./betas/pubmed_CombinedModel_{models[i]}_clean_{attack_type}_.log").tolist()]
            beta_values["perturb_0.1"] = [
                np.loadtxt(f"./betas/pubmed_CombinedModel_{models[i]}_{attack_type}_0.1.log").tolist()]
            beta_values["perturb_0.2"] = [
                np.loadtxt(f"./betas/pubmed_CombinedModel_{models[i]}_{attack_type}_0.2.log").tolist()]

        labels = beta_values.keys()
        epochs = list(range(1, num_epochs[i] + 1))  # List of epoch numbers

        # Plot each beta value trajectory
        for label in labels:
            line, = ax.plot(epochs, beta_values[label][0][:num_epochs[i]], label=label, linewidth=2)

            # Collect unique lines and labels for each attack type (only add once per label)
            if j == 0 and label not in labels_nettack:
                lines_nettack.append(line)
                labels_nettack.append(label)
            elif j == 1 and label not in labels_meta:
                lines_meta.append(line)
                labels_meta.append(label)

        ax.grid(True)

# Place legends inside the first subplot of each column for each attack type
axes[2, 0].legend(lines_nettack,
                  labels_nettack,
                  title="Nettack Legend",
                  loc="lower left",
                  frameon=True,
                  fontsize=12,
                  title_fontsize=12)
axes[2, 1].legend(lines_meta,
                  labels_meta,
                  title="Meta Legend",
                  loc="lower left",
                  frameon=True,
                  fontsize=12,
                  title_fontsize=12)

# Add global x and y labels for the entire figure
fig.text(0.5, -0.01, 'Number of Epochs', ha='center', va='center', fontsize=14, fontweight='semibold')
fig.text(0.01, 0.5, 'Beta Value', ha='center', va='center', rotation='vertical', fontsize=14, fontweight='semibold')

# Adjust layout
plt.tight_layout()
plt.savefig("beta_values_plot_pubmed_all.png", dpi=600, bbox_inches='tight')
"""
