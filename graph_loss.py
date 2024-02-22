import os
import json
import matplotlib.pyplot as plt

def plot_losses(folder_path, title):
    all_loss_g_values = []
    all_loss_d_values = []
    all_iterations = []
    last_iteration = 0
    lines_to_skip = 20 # No. outliers (caused by resuming training) to remove for smoothing
    is_first_file = True

    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(folder_path, file_name)

            loss_g_values = []
            loss_d_values = []
            iterations = []

            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    if not is_first_file and line_num < lines_to_skip:
                        continue # Skip outlier from resuming training (transfer learning)

                    data = json.loads(line)
                    progress_kimg = data.get("Progress/kimg", {}).get("mean", None)
                    loss_g = data.get("Loss/G/loss", {}).get("mean", None)
                    loss_d = data.get("Loss/D/loss", {}).get("mean", None)

                    if loss_g is not None and loss_d is not None:
                        iterations.append(progress_kimg + last_iteration)
                        loss_g_values.append(loss_g)
                        loss_d_values.append(loss_d)

            last_iteration = max(iterations)
            all_iterations.extend(iterations)
            all_loss_g_values.extend(loss_g_values)
            all_loss_d_values.extend(loss_d_values)
            is_first_file = False

    plt.plot(all_iterations, all_loss_g_values, label='Generator')
    plt.plot(all_iterations, all_loss_d_values, label='Discriminator')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

folder_path = '/path/to/stats/folder'
title = r"Title For Loss Graph"
plot_losses(folder_path, title)