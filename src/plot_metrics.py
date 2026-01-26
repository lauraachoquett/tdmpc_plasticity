import matplotlib.pyplot as plt
import os 
import csv 
import numpy as np
import cv2
import os
import re

def plot_metrics(folder: str):
    log_path = os.path.join(folder, "train.log")
    log_path_eval = os.path.join(folder, "eval.log")

    env_step_rewards= []
    env_step_metrics = []
    episode_reward = []
    grad_norm = []
    weight_distance = []
    weight_magnitude = []
    zgr = []
    fzar = []
    srank = []
    

    required_fields = [
        "env_step",
        "episode_reward",
        "grad_norm",
        "weight_distance",
        "weight_magnitude",
        "zgr",
        "fzar",
        "srank",
    ]

    skipped = 0

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # vérifier que toutes les valeurs sont présentes et non vides
            env_step_rewards.append(int(row["env_step"]))
            episode_reward.append(float(row["episode_reward"]))
            if any(row[field] == "" for field in required_fields):
                skipped += 1
                continue

            try:
                env_step_metrics.append(int(row["env_step"]))
                grad_norm.append(float(row["grad_norm"]))
                weight_distance.append(float(row["weight_distance"]))
                weight_magnitude.append(float(row["weight_magnitude"]))
                zgr.append(float(row["zgr"]))
                fzar.append(float(row["fzar"]))
                srank.append(float(row["srank"]))
            except ValueError:
                skipped += 1
                continue

    eval_rewards = []
    env_step_eval = []
    
    with open(log_path_eval, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            env_step_eval.append(float(row["env_step"]))
            eval_rewards.append(float(row["episode_reward"]))
    

            


    # Création figure 2x2
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs = axs.flatten()  # pour indexer facilement axs[0], axs[1], ...

    # 1️⃣ Weight magnitude & distance
    axs[0].plot(env_step_metrics, weight_magnitude, label='Weight Magnitude', color='tab:blue')
    axs[0].plot(env_step_metrics, weight_distance, label='Weight Distance', color='tab:orange')
    axs[0].set_title('Weights Metrics')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Value')
    axs[0].legend()
    axs[0].grid(True)

    # 2️⃣ ZGR
    axs[1].plot(env_step_metrics, np.array(zgr)*100, color='tab:red')
    axs[1].set_title('Zero Gradient Ratio (ZGR)')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('ZGR (%)')
    axs[1].grid(True)

    # 3️⃣ FZAR
    axs[2].plot(env_step_metrics, np.array(fzar)*100, color='tab:green')
    axs[2].set_title('Feature Zero Activation Ratio (FZAR)%')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('FZAR (%)')
    axs[2].grid(True)

    # 4️⃣ Feature Rank
    axs[3].scatter(env_step_metrics, srank, color='tab:purple')
    axs[3].set_title('Feature Rank (srank)')
    axs[3].set_xlabel('Episode')
    axs[3].set_ylabel('srank')
    axs[3].grid(True)
    
    axs[4].plot(env_step_metrics,grad_norm,color='orange')
    axs[4].set_title('Gradient Norm (gn)')
    axs[4].set_xlabel('Episode')
    axs[4].set_ylabel('gn')
    axs[4].grid(True) 
    
    axs[5].plot(env_step_rewards, episode_reward, label='training reward', color='tab:blue')
    axs[5].plot(env_step_eval, eval_rewards, label='evaluation reward', color='tab:red')
    axs[5].set_title('Training and Evaluation Reward')
    axs[5].set_xlabel('Episode')
    axs[5].set_ylabel('Reward')
    axs[5].legend()
    axs[5].grid(True)

    plt.tight_layout()
    path_save_fig = os.path.join(folder,'fig/')
    os.makedirs(path_save_fig,exist_ok=True)
    plt.savefig(path_save_fig +'plot_metrics.png',dpi=300,bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(env_step_rewards, episode_reward, label='training reward', color='tab:blue')
    ax.plot(env_step_eval, eval_rewards, label='evaluation reward', color='tab:red')
    ax.set_title('Training and Evaluation Reward')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True)
    plt.savefig(path_save_fig +'plot_reward.png',dpi=300,bbox_inches='tight')

def save_K(K,save_file,step):
    folder_save_K = os.path.join(save_file,'data')
    os.makedirs(folder_save_K,exist_ok=True)
    K_np = K.cpu().numpy()
    path_save_K = f'{folder_save_K}/K_{step}'
    np.save(path_save_K,K_np)
    return folder_save_K

def plot_K(folder_data,save_file,step):
    path_save_K = f'{folder_data}/K_{step}.npy'
    K_np = np.load(path_save_K)
    # K_norm = K_np / np.max(np.abs(K_np))
    K_norm = K_np
    plt.figure(figsize=(6,6))
    plt.imshow(K_norm, cmap='coolwarm', aspect='auto',vmin=-1, vmax=1,)
    plt.colorbar(label='Normalized K values')
    plt.title("Heatmap of NTK / Gradient Covariance Matrix")
    plt.xlabel("Input index")
    plt.ylabel("Input index")
    plt.tight_layout()
    path_save_file = os.path.join(save_file,'fig/')
    os.makedirs(path_save_file,exist_ok=True)
    plt.savefig(f'{path_save_file}/K_{step}.png')


from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt

def create_ntk_video_matplotlib(save_file, output_name='ntk_evolution.mp4', fps=1,repeat_frames=10):
    fig_path = os.path.join(save_file, 'fig/')
    files = sorted([f for f in os.listdir(fig_path) if f.startswith('K_')],
                   key=lambda x: int(re.search(r'K_(\d+)', x).group(1)))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    writer = FFMpegWriter(fps=fps)
    video_path = os.path.join(save_file, output_name)
    
    with writer.saving(fig, video_path, dpi=300):
        for file in files:
            step = int(re.search(r'K_(\d+)', file).group(1))
            img = plt.imread(os.path.join(fig_path, file))
            
            ax.clear()
            ax.imshow(img)
            ax.set_title(f'NTK at step {step}')
            ax.axis('off')
            for _ in range(repeat_frames):
                writer.grab_frame()
    
    plt.close()
    print(f"Vidéo sauvegardée : {video_path}")



    
if __name__ == '__main__' : 
    path_train = 'logs/cartpole-swingup/state/K_BASIC_TDMPC/1' 
    plot_metrics(path_train)
    # save_file = 'logs/cartpole-swingup/state/K_BASIC_TDMPC/1'
    # create_ntk_video_matplotlib(save_file, output_name='ntk_evolution.mp4', fps=5)