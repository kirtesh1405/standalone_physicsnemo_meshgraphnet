"""Small, explicit wrapper around the maintained PhysicsNeMo MeshGraphNet."""

from torch import nn


class PhysicsNemoMeshGraphNet(nn.Module):
    """Predict four normalized vertex fields from mesh and load features."""

    def __init__(
        self,
        *,
        input_dim: int = 14,
        edge_dim: int = 4,
        output_dim: int = 4,
        hidden_dim: int = 48,
        processor_size: int = 6,
        mlp_layers: int = 2,
    ) -> None:
        super().__init__()
        from physicsnemo.models.meshgraphnet import MeshGraphNet

        self.network = MeshGraphNet(
            input_dim_nodes=input_dim,
            input_dim_edges=edge_dim,
            output_dim=output_dim,
            processor_size=processor_size,
            hidden_dim_processor=hidden_dim,
            num_layers_node_processor=mlp_layers,
            num_layers_edge_processor=mlp_layers,
        )

    def forward(self, node_features, edge_features, graph):
        return self.network(node_features, edge_features, graph)
