import rich
from enum import Enum, auto
import os
from networkx.drawing.nx_agraph import to_agraph
import networkx as nx
from typing import Optional, Sequence, Union
from textx import metamodel_from_file
import click
from PIL import Image


class BaseNode:
    NODES_MAP = {}

    def __init__(self, name: str) -> None:
        self._base_name = name
        BaseNode.NODES_MAP[self._base_name] = self

    def __repr__(self) -> str:
        return self._base_name

    def __hash__(self) -> int:
        return hash(self._base_name)

    def __eq__(self, o: object) -> bool:
        return self._base_name == o._base_name


class PortType(Enum):
    INPUT_PORT = auto()
    OUTPUT_PORT = auto()


class DataNode(BaseNode):

    def __init__(self,
                 block_name: str,
                 port_type: PortType,
                 name: str,
                 item: Optional[Union[str, int]] = None
                 ) -> None:

        self.block_name = block_name
        self.name = name
        self.item = item
        self.port_type = port_type
        self._full_name = f'{block_name}#{port_type.name}#{name}' + (f'[{item}]' if item else '')
        super().__init__(name=self._full_name)

    def is_index_node(self):
        return self.item is not None

    def to_string(self, with_block: bool = False):
        s = f'{self.block_name}.' if with_block else ''
        s += self.name
        if self.is_index_node():
            s += f'[{self.item}]'
        return s


class BlockNode(BaseNode):

    def __init__(self, name: str) -> None:
        super().__init__(name=name)


class Connection:

    def __init__(self,
                 parent,
                 node_start,
                 port_start,
                 port_start_arity,
                 node_end,
                 port_end,
                 port_end_arity
                 ):
        self.parent = parent
        self.node_start = node_start
        self.port_start = port_start
        self.port_start_arity = port_start_arity
        self.node_end = node_end
        self.port_end = port_end
        self.port_end_arity = port_end_arity

    def get_block(self, node, port, arity, ptype: PortType):
        block = BlockNode(name=node.name)
        port = DataNode(
            block_name=node.name,
            port_type=ptype,
            name=port.name,
            item=None
        )
        port_item = None
        if arity:
            port_item = DataNode(
                block_name=node.name,
                port_type=ptype,
                name=port.name,
                item=arity.idx if arity.idx else arity.name
            )
        return block, port, port_item

    def get_start_block(self):
        return self.get_block(
            node=self.node_start,
            port=self.port_start,
            arity=self.port_start_arity,
            ptype=PortType.OUTPUT_PORT
        )

    def get_end_block(self):
        return self.get_block(
            node=self.node_end,
            port=self.port_end,
            arity=self.port_end_arity,
            ptype=PortType.INPUT_PORT
        )

    def connect(self, graph: nx.DiGraph):
        block_start, port_start, port_item_start = self.get_start_block()
        block_end, port_end, port_item_end = self.get_end_block()

        graph.add_edge(block_start, port_start)

        _n = port_start
        if port_item_start:
            graph.add_edge(port_start, port_item_start)
            _n = port_item_start

        if port_item_end:
            graph.add_edge(_n, port_item_end)
            graph.add_edge(port_item_end, port_end)
        else:
            graph.add_edge(_n, port_end)
        graph.add_edge(port_end, block_end)


@click.command('parse_graph')
@click.option('-m', '--model_filename', default='model.tx')
@click.option('-i', '--input_filename', default='dataset_pipeline.g')
@click.option('-o', '--output_image', default='/tmp/output.png')
@click.option('--layout', default='dot', type=click.Choice('neato|dot|twopi|circo|fdp|nop'.split('|')))
def parse_graph(model_filename, input_filename, output_image, layout):

    # Load Metamodel from textX file
    metamodel = metamodel_from_file(
        model_filename,
        classes=[Connection]
    )

    # Create Model from Metamodel and target file
    model = metamodel.model_from_file(
        input_filename
    )

    # Create D-Graph
    g = nx.DiGraph()
    for connection in model.connection:
        connection.connect(g)

    # Draw Graph

    A = to_agraph(g)
    # for i, graph_edge in enumerate(A.iteredges()):
    # print(i, type(graph_edge))
    # graph_edge.attr['arrowhead'] = 'dot'
    for i, graph_node in enumerate(A.iternodes()):
        user_node = BaseNode.NODES_MAP[str(graph_node)]
        graph_node.attr['color'] = '#ffffff'
        if isinstance(user_node, BlockNode):
            graph_node.attr['label'] = str(user_node)
            graph_node.attr['style'] = 'filled'
            graph_node.attr['fillcolor'] = '#26a69a'
            graph_node.attr['fontcolor'] = '#fafafa'
            graph_node.attr['fontsize'] = 52
            graph_node.attr['shape'] = 'box3d'
        if isinstance(user_node, DataNode):
            if user_node.is_index_node():
                graph_node.attr['label'] = str(user_node.item)
                graph_node.attr['style'] = 'filled'
                if user_node.port_type == PortType.INPUT_PORT:
                    graph_node.attr['fillcolor'] = '#a5d6a7'
                else:
                    graph_node.attr['fillcolor'] = '#ef9a9a'
                graph_node.attr['fontcolor'] = '#222222'
                graph_node.attr['fontsize'] = 16
                graph_node.attr['shape'] = 'underline'
                graph_node.attr['color'] = '#222222'
            else:
                if user_node.port_type == PortType.INPUT_PORT:
                    graph_node.attr['fillcolor'] = '#4caf50'
                    graph_node.attr['shape'] = 'box'
                else:
                    graph_node.attr['fillcolor'] = '#f44336'
                    graph_node.attr['shape'] = 'box'

                graph_node.attr['label'] = str(user_node.name)
                graph_node.attr['style'] = 'filled'
                graph_node.attr['fontcolor'] = '#fafafa'
                graph_node.attr['fontsize'] = 16

    A.layout(layout)
    A.draw(output_image)

    image = Image.open(output_image)
    image.show("Graph")

    # 💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩
    # 💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩
    # 💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩
    # Debug parse, from here on the code is ready for 'https://shitcode.net/'
    # 💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩
    # 💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩
    # 💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩

    def get_real_input_data_node(node):
        input_nodes = [x[0] for x in g.in_edges(node)]
        outputs = []
        for i in input_nodes:
            if i.block_name == node.block_name:
                outputs.append(i)
        return [node] if len(outputs) == 0 else outputs

    def get_real_output_data_node(node):
        output_nodes = [x[0] for x in g.out_edges(node)]
        outputs = []
        for o in output_nodes:
            if o.block_name == node.block_name:
                outputs.append(o)
        return [node] if len(outputs) == 0 else outputs

    def get_previous_output(node):
        input_nodes = [x[0] for x in g.in_edges(node)]
        assert len(input_nodes) == 1
        return input_nodes[0]

    for node in g.nodes():
        if isinstance(node, BlockNode):
            rich.print(f"[green]Block[/green] [u]{node}")
            out_edges = g.out_edges(node)

            # Inputs
            rich.print("\t[green]Input:")
            input_nodes = [x[0] for x in g.in_edges(node)]
            for input_node in input_nodes:
                n: Sequence[DataNode] = get_real_input_data_node(input_node)
                n_back: Sequence[DataNode] = [get_previous_output(x) for x in n]

                for idx in range(len(n)):
                    rich.print("\t\t", n[idx].to_string(), '=', n_back[idx].to_string(with_block=True))

            # Outputs
            rich.print("\t[green]Output:")
            output_nodes = [x[1] for x in g.out_edges(node)]
            for output_node in output_nodes:
                n: Sequence[DataNode] = get_real_output_data_node(output_node)
                for idx in range(len(n)):
                    rich.print("\t\t", n[idx].to_string())


if __name__ == '__main__':
    parse_graph()
