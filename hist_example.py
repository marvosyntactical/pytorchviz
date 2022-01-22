import torch
from torch import nn

from torchviz import HistManager


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()

        self.max_pool2d = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten(1)

        self.fc1 = nn.Linear(9216, 128)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 128)
        self.relu4 = nn.ReLU()
        self.residual = nn.Identity()

        self.fc3 = nn.Linear(128, 10)
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        x = self.max_pool2d(x)
        x = self.dropout1(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout2(x)

        skip = x
        x = self.fc2(x)
        x = self.relu4(x)

        x = x.clone() + skip # residual
        x = self.residual(x)

        x = self.fc3(x)
        output = self.log_softmax(x)

        return output


def main():
    # HistManager API example

    plot_directory = "tmp_plots"

    N = 2
    freq = 2
    indices = [1,2,3]

    m = Net()

    # init histogram manager
    HM = HistManager(
        name="toy",
        hist_freq=freq,
        hist_indices=indices,
        bins=200,
        hist_superdir=plot_directory,
        record_input_dists=False,
        record_output_dists=True,
        verbose=False
    )

    # let it process our model
    HM.process_model(m)

    # train normally
    get_a_batch = lambda: torch.randn(10, 1, 28, 28)

    for i in range(N):
        data = get_a_batch()
        y = m(data)
        y.sum().backward()

    # plot an activation graph using the latest histograms
    data = get_a_batch()

    HM.plot_model_graph(
        input_data=data,
    )

    # leave the model alone now.
    HM.stop()

    return


if __name__ == "__main__":
    main()
