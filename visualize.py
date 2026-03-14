import os
import torch
import argparse
import numpy as np
from matplotlib import pyplot as plt

from utils.data_utils import set_seed, str2bool, assign_colors
from nets.attention_model import AttentionModel
from utils.functions import load_problem
from utils import load_model


def arguments(args=None):
    parser = argparse.ArgumentParser(description="Visualize predictions made by the trained attention model")
    parser.add_argument('--seed', type=int, default=0, help='Random seed to use')

    # Model
    parser.add_argument('--model', help='Path to a trained attention model checkpoint or checkpoint directory')

    # Problem
    parser.add_argument('--problem', default='top', help="The problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--data_distribution', type=str, default='const',
                        help='Data distribution to use during training, defaults and options depend on problem')
    parser.add_argument('--num_agents', type=int, default=2, help="Number of agents")
    parser.add_argument('--num_depots', type=int, default=1, help="Number of depots. Options are 1 or 2. num_depots=1 "
                        "means that the start and end depot are the same. num_depots=2 means that they are different")
    parser.add_argument('--max_length', type=float, default=2, help="Normalized time limit to solve the problem")

    # Data loading
    parser.add_argument('--dataset_file', type=str, default=None, help='Path to dataset pkl to load instance from')
    parser.add_argument('--instance_idx', type=int, default=0, help='Index of instance to visualize from dataset (default 0)')

    # Output
    parser.add_argument('--output', type=str, default=None, help='Output path for saved figure (default: outputs/tour.png)')

    # CPU / GPU
    parser.add_argument('--use_cuda', type=str2bool, default=True, help="True to use CUDA")
    opts = parser.parse_args(args)
    opts.use_cuda = torch.cuda.is_available() and opts.use_cuda

    # Check problem is correct
    assert opts.problem == 'top', "Only the top problem is supported"
    assert opts.num_agents > 0, 'num_agents must be greater than 0'

    # Check model path exists
    assert opts.model is not None and os.path.exists(opts.model), 'Path to attention model does not exist'
    return opts


def plot_tour(tours, inputs, problem, model_name, data_dist='', output_path='outputs/tour.png'):
    """
    Plot a given tour.
    # Arguments
        tours (numpy array): contains one ordered list of nodes per agent.
        inputs (dict or numpy array): if TSP, inputs is an array containing the coordinates of the nodes. Otherwise, it
        is a dict with the coordinates of the nodes (loc) and the depot (depot), and other possible features.
        problem (str): name of the problem.
        model_name (str): name of the model.
        data_dist (str): type of prizes for the OP. For any other problem, just set this to ''.
    """

    # Number of agents
    num_agents = len(tours)
    colors = assign_colors(num_agents) if num_agents <= 6 else np.random.rand(num_agents, 3)
    # colors = ['tab:green', 'tab:blue', 'tab:orange', 'tab:pink', 'tab:purple']

    # Initialize plot
    fig, ax = plt.subplots()
    plt.xticks([])
    plt.yticks([])
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    # Depot (blue circle)
    depot = inputs['depot']
    plt.scatter(depot[0], depot[1], c='b')
    depot2 = inputs['depot2'] if 'depot2' in inputs else depot
    plt.scatter(depot2[0], depot2[1], c='r')  # (red circle)

    # Nodes (black circles)
    loc = inputs['loc']
    plt.scatter(loc[..., 0], loc[..., 1], c='k')
    loc = np.concatenate(([depot], loc, [depot2]), axis=0)

    # Prizes (add prize 0 to depots)
    if len(inputs['prize']) == len(loc):
        prizes = inputs['prize']
    else:
        prizes = np.concatenate(([0], inputs['prize'], [0]), axis=0)

    # For each agent
    reward, length = 0, 0
    for k, tour in enumerate(tours):

        # Calculate the length of the tour
        nodes = np.take(loc, tour, axis=0)
        d = np.sum(np.linalg.norm(nodes[1:] - nodes[:-1], axis=1))
        length = length if length >= d else d
        reward += np.sum(np.take(prizes, tour))

        # Draw arrows
        for i in range(1, tour.shape[0]):
            dx = loc[tour[i], 0] - loc[tour[i - 1], 0]
            dy = loc[tour[i], 1] - loc[tour[i - 1], 1]
            plt.arrow(loc[tour[i - 1], 0], loc[tour[i - 1], 1], dx, dy, head_width=.025, fc=colors[k], ec=None,
                      length_includes_head=True)

    # plt.text(0.02, 0.95, f"Total reward: {reward:.2f}", transform=ax.transAxes,
    #          fontsize=12, fontweight='bold', ha='left', va='top')

    # Save to specified output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Saved] {output_path}")
    plt.show()


def reshape_tours(tours, num_agents, end_ids=0):
    new_tours = [[] for _ in range(num_agents)]
    count, check = 0, True
    for node in tours.reshape(-1, order='F'):
        if count >= num_agents:
            break
        if node == end_ids:
            if check:
                count += 1
            check = False
        else:
            new_tours[count].append(node)
            check = True
    return new_tours


def add_depots(tours, num_agents, graph_size):
    tours = list(tours)
    for k in range(num_agents):
        tours[k] = np.array(tours[k])
        if len(tours[k]) > 0:
            if tours[k][0] != 0:
                tours[k] = np.concatenate(([0], tours[k]), axis=0)
            if tours[k][-1] != graph_size + 1:
                tours[k] = np.concatenate((tours[k], [graph_size + 1]), axis=0)
        else:
            tours[k] = np.array([0, graph_size + 1])
        print('Agent {}: '.format(k + 1), tours[k])
    return tours


def main(opts):

    # Set seed for reproducibility
    set_seed(opts.seed)

    # Load problem
    problem = load_problem(opts.problem)
    
    # Load dataset: from file or generate
    if opts.dataset_file is not None:
        dataset = problem.make_dataset(
            filename=opts.dataset_file, 
            num_samples=opts.instance_idx + 1,
            num_agents=opts.num_agents, 
            num_depots=opts.num_depots
        )
        inputs = dataset.data[opts.instance_idx]
    else:
        dataset = problem.make_dataset(
            size=opts.graph_size, 
            num_samples=1, 
            distribution=opts.data_distribution,
            max_length=opts.max_length, 
            num_agents=opts.num_agents, 
            num_depots=opts.num_depots
        )
        inputs = dataset.data[0]

    # Set the device
    device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Load attention model for evaluation on the chosen device
    model, model_args = load_model(opts.model, num_agents=opts.num_agents)
    model.set_decode_type('greedy')
    model.num_depots = opts.num_depots
    model.num_agents = opts.num_agents
    model.eval()  # Put in evaluation mode to not track gradients
    model.to(device)

    # Auto-detect model name based on args.json
    assert isinstance(model, AttentionModel), 'Model should be an instance of AttentionModel'
    has_rope = model_args.get('use_rope', False)
    has_entropy = model_args.get('use_entropy_scheduling', False)
    if has_rope and has_entropy:
        model_name = 'MARGA'
    elif has_rope:
        model_name = 'RGA'
    elif has_entropy:
        model_name = 'Former-ESD'
    else:
        model_name = 'TOP-Former'

    # Calculate tour
    for k, v in inputs.items():
        inputs[k] = v.unsqueeze(0).to(device)
    _, _, tours = model(inputs, return_pi=True)

    # Torch tensors to numpy
    tours = tours.cpu().detach().numpy().squeeze()
    for k, v in inputs.items():
        inputs[k] = v.cpu().detach().numpy().squeeze()

    # Reshape tours list
    tours = reshape_tours(tours, opts.num_agents, end_ids=inputs['loc'].shape[0] + 1)

    # Add depots and print tours (use actual graph size from data to avoid mismatch)
    tours = add_depots(tours, opts.num_agents, inputs['loc'].shape[0])

    # Plot tours with custom output path
    output_path = opts.output if opts.output else 'outputs/tour.png'
    plot_tour(tours, inputs, problem.NAME, model_name, data_dist=opts.data_distribution, output_path=output_path)

    # Add black border to saved image
    from PIL import Image, ImageOps
    img = Image.open(output_path)
    bordered = ImageOps.expand(img, border=5, fill='black')
    bordered.save(output_path)



if __name__ == "__main__":
    main(arguments())
