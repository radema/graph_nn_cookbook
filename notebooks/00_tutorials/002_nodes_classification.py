import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import copy
    import os
    import sys

    import marimo as mo
    import torch
    import torch.nn.functional as F
    from torch.nn import Linear
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import GATConv, GCNConv
    from torch_geometric.transforms import NormalizeFeatures

    sys.path.append(os.path.abspath("../src"))

    from viz_utils import visualize_TSNE

    return (
        F,
        GATConv,
        GCNConv,
        Linear,
        NormalizeFeatures,
        Planetoid,
        copy,
        mo,
        torch,
        visualize_TSNE,
    )


@app.cell
def _(mo):
    mo.md(
        r"""This notebook is based on the PyG's tutorial [Node Classification Notebook](https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing#scrollTo=imGrKO5YH11-). It has been completed based on the suggested exercises."""  # noqa:E501
    )
    return


@app.cell
def _(NormalizeFeatures, Planetoid):
    dataset = Planetoid(
        root="data/Planetoid", name="Cora", transform=NormalizeFeatures()
    )

    print()
    print(f"Dataset: {dataset}: ")
    print("======================")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    return (dataset,)


@app.cell
def _(dataset):
    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print(
        "==========================================================================================================="  # noqa:E501
    )

    # Gather some statistics about the graph.
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")  # noqa: E231
    print(f"Number of training nodes: {data.train_mask.sum()}")
    print(
        f"Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}"  # noqa: E231
    )
    print(f"Has isolated nodes: {data.has_isolated_nodes()}")
    print(f"Has self-loops: {data.has_self_loops()}")
    print(f"Is undirected: {data.is_undirected()}")
    return (data,)


@app.cell
def _(F, Linear, dataset, torch):
    class MLP(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            torch.manual_seed(12345)
            self.lin1 = Linear(dataset.num_features, hidden_channels)
            self.lin2 = Linear(hidden_channels, dataset.num_classes)

        def forward(self, x):
            x = self.lin1(x)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin2(x)
            return x

    return (MLP,)


@app.cell
def _(F, GCNConv, torch):
    class GCN(torch.nn.Module):
        def __init__(
            self,
            num_features: int,
            num_classes: int,
            num_layers: int = 2,
            hidden_channels: int = 16,
        ):
            super().__init__()
            torch.manual_seed(1234567)
            self.convs = torch.nn.ModuleList()

            # Input layer
            self.convs.append(GCNConv(num_features, hidden_channels))

            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))

            # Output layer
            self.convs.append(GCNConv(hidden_channels, num_classes))

        def forward(self, x, edge_index):
            for conv in self.convs[:-1]:
                x = conv(x, edge_index)
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
            x = self.convs[-1](x, edge_index)
            return x

    return (GCN,)


@app.cell
def _(F, GATConv, dataset, torch):
    class GAT(torch.nn.Module):
        def __init__(self, hidden_channels, heads):
            super().__init__()
            torch.manual_seed(1234567)
            self.conv1 = GATConv(
                in_channels=dataset.num_features,
                out_channels=hidden_channels,
                heads=heads,
            )
            self.conv2 = GATConv(
                in_channels=hidden_channels * heads,
                out_channels=dataset.num_classes,
                heads=1,
                concat=False,
            )

        def forward(self, x, edge_index):
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)
            return x

    return (GAT,)


@app.cell
def _(mo):
    HIDDEN_CHANNELS_INPUT = mo.ui.number(
        start=1, value=16, step=1, stop=100, label="Hidden Channels: "
    )
    HIDDEN_CHANNELS_INPUT
    return (HIDDEN_CHANNELS_INPUT,)


@app.cell
def _(mo):
    NUM_LAYERS_INPUT = mo.ui.number(
        start=1, value=2, step=1, stop=100, label="Number of Layers: "
    )
    NUM_LAYERS_INPUT
    return (NUM_LAYERS_INPUT,)


@app.cell
def _(mo):
    NUMBER_EPOCHS_INPUT = mo.ui.number(
        start=1, value=200, step=1, stop=2000, label="Number of epochs: "
    )
    NUMBER_EPOCHS_INPUT
    return (NUMBER_EPOCHS_INPUT,)


@app.cell
def _(
    GCN,
    HIDDEN_CHANNELS_INPUT,
    NUM_LAYERS_INPUT,
    data,
    dataset,
    visualize_TSNE,
):
    model_gcn = GCN(
        hidden_channels=HIDDEN_CHANNELS_INPUT.value,
        num_layers=NUM_LAYERS_INPUT.value,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
    )
    model_gcn.eval()

    out = model_gcn(data.x, data.edge_index)
    visualize_TSNE(out, color=data.y)
    return


@app.cell
def _(
    GAT,
    GCN,
    HIDDEN_CHANNELS_INPUT,
    MLP,
    NUMBER_EPOCHS_INPUT,
    NUM_LAYERS_INPUT,
    copy,
    data,
    dataset,
    torch,
    visualize_TSNE,
):
    print("MLP model")
    model = MLP(hidden_channels=HIDDEN_CHANNELS_INPUT.value)
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4
    )  # Define optimizer.

    def train():
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x)  # Perform a single forward pass.
        loss = criterion(
            out[data.train_mask], data.y[data.train_mask]
        )  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def test():
        model.eval()
        out = model(data.x)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = (
            pred[data.test_mask] == data.y[data.test_mask]
        )  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(
            data.test_mask.sum()
        )  # Derive ratio of correct predictions.
        return test_acc

    for epoch in range(1, NUMBER_EPOCHS_INPUT.value):
        loss = train()
        if epoch % 40 == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")  # noqa: E231
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")  # noqa: E231
    test_acc = test()
    print(f"Test Accuracy: {test_acc:.4f}")  # noqa: E231

    print("=" * 20)
    print("GCN model")
    model_gcnn = GCN(
        hidden_channels=HIDDEN_CHANNELS_INPUT.value,
        num_layers=NUM_LAYERS_INPUT.value,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
    )
    optimizer = torch.optim.Adam(model_gcnn.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = None

    def train():
        model_gcnn.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model_gcnn(data.x, data.edge_index)  # Perform a single forward pass.
        loss = criterion(
            out[data.train_mask], data.y[data.train_mask]
        )  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def evaluate():
        model_gcnn.eval()
        out = model_gcnn(data.x, data.edge_index)
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_pred = out[data.val_mask].argmax(dim=1)
        val_acc = (
            val_pred == data.y[data.val_mask]
        ).sum().item() / data.val_mask.sum().item()
        return val_loss.item(), val_acc

    def test():
        model_gcnn.eval()
        out = model_gcnn(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = (
            pred[data.test_mask] == data.y[data.test_mask]
        )  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(
            data.test_mask.sum()
        )  # Derive ratio of correct predictions.
        return test_acc

    for epoch in range(1, NUMBER_EPOCHS_INPUT.value):
        loss = train()
        val_loss, val_acc = evaluate()

        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model_gcnn.state_dict())

        if epoch % 40 == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
    # Load best model based on validation loss
    model_gcnn.load_state_dict(best_model_state)

    # Final test
    test_acc = test()
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best epoch: {best_epoch:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    print("=" * 20)
    out_best_model = model_gcnn(data.x, data.edge_index)
    visualize_TSNE(out_best_model, color=data.y)

    print("=" * 20)
    print("GAT model")
    model_gat = GAT(hidden_channels=8, heads=8)

    optimizer = torch.optim.Adam(model_gat.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = None

    def train():
        model_gat.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model_gat(data.x, data.edge_index)  # Perform a single forward pass.
        loss = criterion(
            out[data.train_mask], data.y[data.train_mask]
        )  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def evaluate():
        model_gat.eval()
        out = model_gat(data.x, data.edge_index)
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_pred = out[data.val_mask].argmax(dim=1)
        val_acc = (
            val_pred == data.y[data.val_mask]
        ).sum().item() / data.val_mask.sum().item()
        return val_loss.item(), val_acc

    def test():
        model_gat.eval()
        out = model_gat(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = (
            pred[data.test_mask] == data.y[data.test_mask]
        )  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(
            data.test_mask.sum()
        )  # Derive ratio of correct predictions.
        return test_acc

    for epoch in range(1, NUMBER_EPOCHS_INPUT.value):
        loss = train()
        val_loss, val_acc = evaluate()

        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model_gat.state_dict())

        if epoch % 40 == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
    # Load best model based on validation loss
    model_gat.load_state_dict(best_model_state)

    # Final test
    test_acc = test()
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best epoch: {best_epoch:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    out_best_model = model_gat(data.x, data.edge_index)
    visualize_TSNE(out_best_model, color=data.y)
    return


if __name__ == "__main__":
    app.run()
