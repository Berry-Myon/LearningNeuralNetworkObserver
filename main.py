import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

# Import config (allow command-line override)
import config.config as config_module

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Network Solver')
    parser.add_argument('--solver_type', type=str, default=None,
                        help='Solver type: Solver_Linear, Solver_NN, Train_Small_NN, Train_Large_NN')
    args = parser.parse_args()

    # Override config with command-line argument if provided
    if args.solver_type:
        config_module.solver_type = args.solver_type

    assert config_module.solver_type in ["Solver_Linear", "Solver_NN", "Train_Small_NN", "Train_Large_NN"], "No such solver type!"

    if config_module.solver_type in ["Solver_Linear", "Solver_NN"]:
        from utils.Solver import Solver
        solver = Solver(config_module.A, config_module.B, config_module.C, config_module.poles, config_module.max_iter, config_module.epsilon)
        solver.solve_short_cut()
        if config_module.solver_type == "Solver_NN":
            solver.solver(config_module.nodes)
        else:
            print("Linear Solver Successful")
    else:
        from train.Two_Stage_Training import Two_Stage_Training
        nn = Two_Stage_Training(config_module.A, config_module.C, config_module.epsilon, config_module.learning_rate, config_module.solver_type, config_module.P, config_module.epochs_pretrain, config_module.epochs_tuning)
        nn.train()
