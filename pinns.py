import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import time
import os
from prettytable import PrettyTable

torch.autograd.set_detect_anomaly(__debug__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if 0 else "cpu"
outfolder = "output"


def count_parameters(model):
    """Counts and prints number of parameters in a PyTorch model"""
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    if 0:
        print(table)
    print(f"Total trainable parameters: {total_params}")
    return total_params


class NeuralNet(nn.Module):
    """Implements a fully connected network"""

    def __init__(
        self,
        input_dimension,
        output_dimension,
        n_hidden_layers,
        neurons,
        regularization_param,
        regularization_exp,
        retrain_seed,
        residual,
    ):
        super(NeuralNet, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_hidden_layers = n_hidden_layers
        self.activation = nn.Tanh()  # nn.SiLU()
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed
        self.residual = residual

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)]
        )
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        self.init_xavier()

    def forward(self, x):
        """The forward function performs the set of affine and non-linear transformations defining the network"""
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            if self.residual:  # and k % 2 == 0:
                # residual connection
                x = self.activation(l(x)) + x
            else:
                x = self.activation(l(x))
        return self.output_layer(x)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_uniform_(m.weight, gain=g)
                # nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = torch.tensor([0]).to(device)
        if self.regularization_param > 0:
            for name, param in self.named_parameters():
                if "weight" in name:
                    reg_loss = reg_loss + torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss


class Pinn:
    """Implements a physics-informed neural network"""

    def __init__(
        self,
        n_int_,
        n_sb_,
        n_tb_,
        t_0_,
        t_L_,
        x_0_,
        x_L_,
        initial_conditions_,
        n_hidden_layers_,
        neurons_,
        regularization_param_,
        regularization_exp_,
        retrain_seed_,
        residual_,
        verbose_,
    ):
        # verbosity
        self.verbose = verbose_

        # extrema of the solution domain (t,x)
        self.domain_extrema = torch.tensor([[t_0_, t_L_], [x_0_, x_L_]])

        # number of spatial dimensions
        self.space_dimensions = self.domain_extrema[1:].shape[0]

        # initial conditions
        self.initial_conditions = initial_conditions_

        # NN to approximate the solution of the PDE
        self.sol = NeuralNet(
            input_dimension=self.domain_extrema.shape[0],
            output_dimension=1,
            n_hidden_layers=n_hidden_layers_,
            neurons=neurons_,
            regularization_param=regularization_param_,
            regularization_exp=regularization_exp_,
            retrain_seed=retrain_seed_,
            residual=residual_,
        ).to(device)
        if self.verbose >= 1:
            count_parameters(self.sol)

        # generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(
            dimension=self.domain_extrema.shape[0]
        )

        # training points
        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_
        self.input_int, self.output_int = None, None
        self.input_tb, self.output_tb = None, None
        self.input_sb, self.output_sb = None, None
        self.sample_training_points()

        # loss history
        self.train_history = list()

        # model file str
        self.model_sol_file = "sol.pt"
        self.model_sol_file = outfolder + "/" + self.model_sol_file

        # helper vars
        self.last_print = 0.0

    def get_parameters(self):
        """Returns all PINN parameters"""
        return list(self.sol.parameters())

    def split_in_half(self, x):
        """Splits a tensor into two equal parts (if possible)"""
        return torch.split(x, x.shape[0] // 2)

    def normalize(self, x):
        """Linearly transforms a tensor whose values are between the domain extrema to a tensor whose values are between 0 and 1"""
        assert x.shape[1] == self.domain_extrema.shape[0]
        xmin = self.domain_extrema[:, 0].to(device)
        xmax = self.domain_extrema[:, 1].to(device)
        return (x - xmin) / (xmax - xmin)

    def denormalize(self, x):
        """Linearly transforms a tensor whose values are between 0 and 1 to a tensor whose values are between the domain extrema"""
        assert x.shape[1] == self.domain_extrema.shape[0]
        xmin = self.domain_extrema[:, 0]
        xmax = self.domain_extrema[:, 1]
        return x * (xmax - xmin) + xmin

    # @torch.compile(fullgraph=True)
    def evaluate_model(self, x):
        """Evaluates the PINN and returns the PDE solution"""
        u = torch.squeeze(self.sol(self.normalize(x.to(device))))
        return u
        ansatz = torch.exp(-((200 * x[:, 0]) ** 2)) * self.initial_conditions(x[:, 1])
        assert u.shape == ansatz.shape
        return ansatz + u

    def add_interior_points(self):
        """Returns the input-output tensor required to assemble the training set corresponding to the interior domain where the PDE is enforced"""
        input_int = self.denormalize(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))  # imposed in loss function
        return input_int.to(device), output_int.to(device)

    def add_temporal_boundary_points(self):
        """Returns the input-output tensor required to assemble the training set corresponding to the temporal boundary"""
        t0 = self.domain_extrema[0, 0]

        input_tb = self.denormalize(self.soboleng.draw(self.n_tb))
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)

        output_tb = self.initial_conditions(input_tb[:, 1])

        return input_tb.to(device), output_tb.to(device)

    def add_spatial_boundary_points(self):
        """Returns the input-output tensor required to assemble the training set corresponding to the spatial boundary"""
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        input_sb = self.denormalize(self.soboleng.draw(self.n_sb))

        input_sb_0 = torch.clone(input_sb)
        input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, xL)

        output_sb_0 = torch.zeros((input_sb.shape[0], 1))  # imposed in loss function
        output_sb_L = torch.zeros((input_sb.shape[0], 1))  # imposed in loss function

        input_sb = torch.cat([input_sb_0, input_sb_L], 0)
        output_sb = torch.cat([output_sb_0, output_sb_L], 0)

        return input_sb.to(device), output_sb.to(device)

    def sample_training_points(self):
        """Samples interior, temporal boundary and spatial boundary training points"""
        self.input_int, self.output_int = self.add_interior_points()
        self.input_tb, self.output_tb = self.add_temporal_boundary_points()
        self.input_sb, self.output_sb = self.add_spatial_boundary_points()

    def compute_loss(self, print_delay=0.0):
        """Computes the total loss function of the problem"""
        self.input_int.requires_grad = True

        # interior loss
        u_int = self.evaluate_model(self.input_int)
        u_int_grad = torch.autograd.grad(
            u_int.sum(), self.input_int, create_graph=True
        )[0]
        u_int_t = u_int_grad[:, 0]
        u_int_x = u_int_grad[:, 1]
        r_int = u_int_t + u_int_x
        assert r_int.shape[0] == self.n_int
        loss_int = torch.mean(abs(r_int) ** 2)

        # temporal boundary loss
        u_tb = self.evaluate_model(self.input_tb)
        r_tb = u_tb - self.output_tb
        assert r_tb.shape[0] == self.n_tb
        loss_tb = 1e2 * torch.mean(abs(r_tb) ** 2)

        # spatial boundary loss (periodic BCs)
        u_sb_0, u_sb_L = self.split_in_half(self.evaluate_model(self.input_sb))
        r_sb = u_sb_0 - u_sb_L
        assert r_sb.shape[0] == self.n_sb
        loss_sb = torch.mean(abs(r_sb) ** 2)

        # regularization loss
        loss_rg = self.sol.regularization()

        # assemble total loss
        loss = torch.log10(loss_int + loss_tb + loss_sb + loss_rg)
        self.train_history.append(loss.item())

        # print stats
        step = len(self.train_history)
        if self.verbose >= 2 and time.time() - self.last_print >= print_delay:
            self.last_print = time.time()
            print(
                "Step: %.5d | Train loss: %+.4f | int: %+.4f tb: %+.4f sb: %+.4f rg: %+.4f"
                % (
                    step,
                    loss.item(),
                    torch.log10(loss_int).item(),
                    torch.log10(loss_tb).item(),
                    torch.log10(loss_sb).item(),
                    torch.log10(loss_rg).item(),
                )
            )

        if step == 9000:
            fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
            im0 = axs[0].scatter(
                self.input_int[:, 1].cpu().detach(),
                self.input_int[:, 0].cpu().detach(),
                c=u_int_t.cpu().detach(),
                cmap="jet",
            )
            plt.colorbar(im0, ax=axs[0])
            im1 = axs[1].scatter(
                self.input_int[:, 1].cpu().detach(),
                self.input_int[:, 0].cpu().detach(),
                c=u_int_x.cpu().detach(),
                cmap="jet",
            )
            plt.colorbar(im1, ax=axs[1])
            plt.tight_layout()
            # plt.show()
            f = outfolder + "/grads.pdf"
            plt.savefig(f)
            if self.verbose >= 1:
                print("Saved " + f)
            plt.close()

            sc = plt.scatter(
                self.input_int[:, 1].cpu().detach(),
                self.input_int[:, 0].cpu().detach(),
                c=(abs(r_int) ** 2).cpu().detach(),
                cmap="jet",
            )
            plt.colorbar(sc)
            plt.tight_layout()
            # plt.show()
            f = outfolder + "/res.pdf"
            plt.savefig(f)
            if self.verbose >= 1:
                print("Saved " + f)
            plt.close()

        return loss

    def fit(self, num_epochs, optimizer, print_delay=0.0):
        """Fits the PINN in order to solve the PDE"""
        for _ in range(num_epochs):

            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(print_delay=print_delay)
                loss.backward()
                return loss

            optimizer.step(closure=closure)

        if self.verbose >= 1 and self.train_history:
            print("Final loss:", self.train_history[-1])

    def plot_training_points(self):
        """Plots the input training points"""
        input_sb = self.add_spatial_boundary_points()[0]
        input_tb = self.add_temporal_boundary_points()[0]
        input_int = self.add_interior_points()[0]

        plt.figure(figsize=(16, 8), dpi=100)
        plt.scatter(
            input_sb[:, 1].cpu().detach().numpy(),
            input_sb[:, 0].cpu().detach().numpy(),
            label="Boundary Points",
        )
        plt.scatter(
            input_int[:, 1].cpu().detach().numpy(),
            input_int[:, 0].cpu().detach().numpy(),
            label="Interior Points",
        )
        plt.scatter(
            input_tb[:, 1].cpu().detach().numpy(),
            input_tb[:, 0].cpu().detach().numpy(),
            label="Initial Points",
        )
        plt.xlabel("$x$")
        plt.ylabel("$t$")
        plt.legend()

        plt.tight_layout()
        # plt.show()
        f = outfolder + "/training_points.pdf"
        plt.savefig(f)
        if self.verbose >= 1:
            print("Saved " + f)
        plt.close()

    def plot_loss(self):
        """Plots the loss in relation to the number of steps"""
        if not self.train_history:
            return
        plt.figure(dpi=100)
        plt.grid(True, which="both", ls=":")
        plt.plot(
            np.arange(1, len(self.train_history) + 1),
            10 ** np.array(self.train_history),
            label="Train Loss",
        )
        plt.yscale("log")
        plt.legend()
        plt.xlabel("Step")
        plt.ylabel("L2 loss")

        plt.tight_layout()
        # plt.show()
        f = outfolder + "/loss.pdf"
        plt.savefig(f)
        if self.verbose >= 1:
            print("Saved " + f)
        plt.close()

    def plot_sol(self):
        """Plots the PINN solution"""
        with torch.no_grad():
            int_points = self.denormalize(self.soboleng.draw(20000))
            u = self.evaluate_model(int_points)

            t0 = self.domain_extrema[0, 0]
            tL = self.domain_extrema[0, 1]
            x0 = self.domain_extrema[1, 0]
            xL = self.domain_extrema[1, 1]
            x = torch.linspace(x0, xL, 2000)
            tb0 = torch.stack((torch.full(x.shape, t0), x), dim=1)
            # tbL = torch.stack((torch.full(x.shape, tL), x), dim=1)
            u0_exact = self.initial_conditions(x)
            u0 = self.evaluate_model(tb0)
            # uL = self.evaluate_model(tbL)

            fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

            im0 = axs[0].scatter(
                x.cpu().detach(), u0_exact.cpu().detach(), label="ICs", marker="o"
            )
            im0 = axs[0].scatter(
                x.cpu().detach(), u0.cpu().detach(), label="PINN", marker="."
            )
            # im0 = axs[0].plot(x.cpu().detach(), uL.cpu().detach(), label="uL")
            axs[0].legend()
            axs[0].set_xlabel("$x$")
            axs[0].set_ylabel("$u$")
            axs[0].grid(True, which="both", ls=":")

            im1 = axs[1].scatter(
                int_points[:, 1].cpu().detach(),
                int_points[:, 0].cpu().detach(),
                c=u.cpu().detach(),
                cmap="jet",
            )
            plt.colorbar(im1, ax=axs[1])
            axs[1].set_xlabel("$x$")
            axs[1].set_ylabel("$t$")
            axs[1].grid(True, which="both", ls=":")

            plt.tight_layout()
            # plt.show()
            f = outfolder + "/sol.pdf"
            plt.savefig(f)
            if self.verbose >= 1:
                print("Saved " + f)
            plt.close()

    def save_model(self):
        """Saves the model to disk"""
        torch.save(self.sol.state_dict(), self.model_sol_file)
        if self.verbose >= 1:
            print("Saved " + self.model_sol_file)

    def load_model(self):
        """Tries to load the model from disk"""
        try:
            sol = torch.load(self.model_sol_file)
            self.sol.load_state_dict(sol)
        except Exception:
            return False
        if self.verbose >= 1:
            print("Loaded " + self.model_sol_file)
        return True


def train():
    """Sets all parameters and solves the PDE"""

    # TODO: what if ICs not periodic? ==> competing residual in bottom corners not a problem?
    # TODO: test loss with linear advection
    # TODO: loss balancing, e.g. with neural network?!

    # settings
    verbose = 2  # verbosity affecting prints, 2
    print_delay = 0.5  # print at most every print_delay seconds, 0.5
    load = 0  # whether to load model from disk, 0

    # hyperparameters, -4.0608720779418945, 43.482420444488525s
    learning_rate = 0.5  # learning rate of the optimizer, 0.5
    history_size = 150  # history size of the optimizer, 150
    n_hidden_layers = 5  # number of hidden layers, 5
    neurons = 50  # number of neurons, 50
    regularization_param = 1e-7  # multiplicative regularization factor, 1e-7
    regularization_exp = 2.0  # regularization norm exponent, 2.0
    retrain_seed = 34  # weight (and bias) init seed, 34
    n_int = 2**11  # number of interior points, 2**11
    n_sb = 2**8  # number of spatial boundary points (per boundary), 2**8
    n_tb = 2**8  # number of temporal boundary points, 2**8
    resample_count = 2  # number of training points resample attempts, 2
    residual = 1  # allow residual connections, 1

    # physics
    t_0, t_L = 0.0, 1.0  # 0.0, 1.0
    x_0, x_L = 0.0, 1.0  # 0.0, 10.0

    def initial_conditions(x):
        """Initial conditions of the PDE"""
        # return torch.where(2 * x % 2 > 1, 0.25, 1)
        # return torch.full(x.shape, 1.53)
        return x
        # return torch.exp(-((x - 0.5) ** 2))
        # return torch.sin(2 * torch.pi * x)
        # return torch.where(2 * x % 2 > 1, 0.25, 1)

    pinn = Pinn(
        n_int,
        n_sb,
        n_tb,
        t_0,
        t_L,
        x_0,
        x_L,
        initial_conditions,
        n_hidden_layers,
        neurons,
        regularization_param,
        regularization_exp,
        retrain_seed,
        residual,
        verbose,
    )

    lbfgs = optim.LBFGS(
        pinn.get_parameters(),
        lr=learning_rate,
        max_iter=int(1e10),
        tolerance_change=1.0 * np.finfo(float).eps,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )

    if not load or not pinn.load_model():
        start = time.time()
        for _ in range(resample_count + 1):
            pinn.fit(1, lbfgs, print_delay=print_delay)
            pinn.sample_training_points()
        end = time.time()

        if pinn.verbose >= 1:
            print("Training time: " + str(end - start) + "s")
        pinn.save_model()

    return pinn


def main():
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    pinn = train()
    pinn.plot_training_points()
    pinn.plot_loss()
    pinn.plot_sol()


if __name__ == "__main__":
    main()
