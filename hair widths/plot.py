import pickle
import matplotlib.pyplot as plt

def plot_saved_hair_widths(dataset_path="all_hairs.pkl"):
    with open(dataset_path, "rb") as f:
        all_hairs = pickle.load(f)
    plt.figure(figsize=(10, 6))
    for i, hair in enumerate(all_hairs):
        width_map = hair['width_map']
        if not width_map:
            continue
        # Use the order along the spine as x-axis
        widths = [w for _, _, w in width_map]
        plt.plot(widths, marker='o', label=f'Hair {i+1}')
    plt.xlabel("Position along spine")
    plt.ylabel("Width (pixels)")
    plt.title("Hair Width Profiles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    plot_saved_hair_widths("all_hairs.pkl")