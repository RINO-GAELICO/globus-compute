import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def process_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract and sort tasks by End Time
    tasks = []
    for task_id, task_data in data.items():
        start_time_str = task_data.get('start_time')
        end_time_str = task_data.get('end_time')
        if start_time_str and end_time_str:
            start_time = datetime.fromtimestamp(float(start_time_str))
            end_time = datetime.fromtimestamp(float(end_time_str))
            tasks.append((end_time, start_time))  # Store (end_time, start_time) tuple

    # Sort tasks by end time
    tasks.sort(key=lambda x: x[0])

    # Convert Times to Numeric Format (seconds since earliest start)
    plot_times = [(end_time - tasks[0][1]).total_seconds() for end_time, _ in tasks]
    plot_counts = list(range(1, len(plot_times) + 1))  # Cumulative count of tasks completed
    
    return plot_times, plot_counts

# Process first file
file_path1 = 'results_pytorch_globus_compute_container_batch_submission_4.json'
plot_times1, plot_counts1 = process_file(file_path1)
plot_times1 = [(x - plot_times1[0]) for x in plot_times1]

# Process second file
file_path2 = 'results_pytorch_globus_compute_container_batch_submission_8.json'
plot_times2, plot_counts2 = process_file(file_path2)
plot_times2 = [(x - plot_times2[0]) for x in plot_times2]

# Process third file
file_path3 = 'results_pytorch_globus_compute_container_batch_submission_12.json'
plot_times3, plot_counts3 = process_file(file_path3)
plot_times3 = [(x - plot_times3[0]) for x in plot_times3]

# Process fourth file
file_path4 = 'results_pytorch_globus_compute_container_concurrent_4.json'
plot_times4, plot_counts4 = process_file(file_path4)
plot_times4 = [(x - plot_times4[0]) for x in plot_times4]

# Process fifth file
file_path5 = 'results_pytorch_globus_compute_container_concurrent_8.json'
plot_times5, plot_counts5 = process_file(file_path5)
plot_times5 = [(x - plot_times5[0]) for x in plot_times5]

# Process sixth file
file_path6 = 'results_pytorch_globus_compute_container_concurrent_12.json'
plot_times6, plot_counts6 = process_file(file_path6)
plot_times6 = [(x - plot_times6[0]) for x in plot_times6]



# Plotting the data for both files in the same graph with normalized x-axis
plt.figure(figsize=(10, 6))

# Plot for the first file (blue)
plt.plot(plot_times1, plot_counts1, marker='o', linestyle='-', color='blue', label='container_batch_submission_4')

# Plot for the second file (red)
plt.plot(plot_times2, plot_counts2, marker='o', linestyle='-', color='red', label='container_batch_submission_8')

# Plot for the third file (green)
plt.plot(plot_times3, plot_counts3, marker='o', linestyle='-', color='green', label='container_batch_submission_12')

# Plot for the fourth file (orange)
plt.plot(plot_times4, plot_counts4, marker='o', linestyle='-', color='orange', label='container_concurrent_4')

# Plot for the fifth file (purple)
plt.plot(plot_times5, plot_counts5, marker='o', linestyle='-', color='purple', label='container_concurrent_8')

# Plot for the sixth file (brown)
plt.plot(plot_times6, plot_counts6, marker='o', linestyle='-', color='brown', label='container_concurrent_12')





plt.xlabel('Time (seconds normalized to earliest start)')
plt.ylabel('Number of Completed Tasks')
plt.title('Tasks Completed Over Time - Batch VS Concurrent')
plt.grid(True)
plt.yticks(range(1, max(len(plot_times3), len(plot_times6)) + 1))
plt.legend()
plt.tight_layout()
# plt.show()

# save in a file called container_VS_venv.png
plt.savefig('Batch_VS_Concurrent.png')

