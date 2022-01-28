from collections import namedtuple
from distutils.version import LooseVersion
import warnings
from typing import Optional
import os

from graphviz import Digraph

import torch
from torch import Tensor
from torch.autograd import Variable

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
from scipy.stats import norm
import pandas as pd
import numpy as np

Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))

# Saved attrs for grad_fn (incl. saved variables) begin with `._saved_*`
SAVED_PREFIX = "_saved_"

def get_fn_name(fn, show_attrs, max_attr_chars):
    name = str(type(fn).__name__)
    if not show_attrs:
        return name
    attrs = dict()
    for attr in dir(fn):
        if not attr.startswith(SAVED_PREFIX):
            continue
        val = getattr(fn, attr)
        attr = attr[len(SAVED_PREFIX):]
        if torch.is_tensor(val):
            attrs[attr] = "[saved tensor]"
        elif isinstance(val, tuple) and any(torch.is_tensor(t) for t in val):
            attrs[attr] = "[saved tensors]"
        else:
            attrs[attr] = str(val)
    if not attrs:
        return name
    max_attr_chars = max(max_attr_chars, 3)
    col1width = max(len(k) for k in attrs.keys())
    col2width = min(max(len(str(v)) for v in attrs.values()), max_attr_chars)
    sep = "-" * max(col1width + col2width + 2, len(name))
    attrstr = '%-' + str(col1width) + 's: %' + str(col2width)+ 's'
    truncate = lambda s: s[:col2width - 3] + "..." if len(s) > col2width else s
    params = '\n'.join(attrstr % (k, truncate(str(v))) for (k, v) in attrs.items())
    return name + '\n' + sep + '\n' + params


def make_dot(var, params=None, show_attrs=False, show_saved=False, max_attr_chars=50, **kwargs):
    """ Produces Graphviz representation of PyTorch autograd graph.

    If a node represents a backward function, it is gray. Otherwise, the node
    represents a tensor and is either blue, orange, or green:
     - Blue: reachable leaf tensors that requires grad (tensors whose `.grad`
         fields will be populated during `.backward()`)
     - Orange: saved tensors of custom autograd functions as well as those
         saved by built-in backward nodes
     - Green: tensor passed in as outputs
     - Dark green: if any output is a view, we represent its base tensor with
         a dark green node.

    Args:
        var: output tensor
        params: dict of (name, tensor) to add names to node that requires grad
        show_attrs: whether to display non-tensor attributes of backward nodes
            (Requires PyTorch version >= 1.9)
        show_saved: whether to display saved tensor nodes that are not by custom
            autograd functions. Saved tensor nodes for custom functions, if
            present, are always displayed. (Requires PyTorch version >= 1.9)
        max_attr_chars: if show_attrs is `True`, sets max number of characters
            to display for any given attribute.
    """
    if LooseVersion(torch.__version__) < LooseVersion("1.9") and \
        (show_attrs or show_saved):
        warnings.warn(
            "make_dot: showing grad_fn attributes and saved variables"
            " requires PyTorch version >= 1.9. (This does NOT apply to"
            " saved tensors saved by custom autograd functions.)")

    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}
    else:
        param_map = {}

    node_attr = dict(
         style='filled',
         shape='box',
         align='left',
         fontsize='10',
         ranksep='0.1',
         height='0.2',
         fontname='monospace',
         image="",
     )
    dot = Digraph(node_attr=node_attr, graph_attr=dict(dpi="384.0"))# size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def get_var_name(var, name=None):
        if not name:
            name = param_map[id(var)] if id(var) in param_map else 'input'
        return '%s\n %s' % (name, size_to_str(var.size()))

    def add_nodes(fn):
        assert not torch.is_tensor(fn)
        if fn in seen:
            return
        seen.add(fn)

        if show_saved:
            for attr in dir(fn):
                if not attr.startswith(SAVED_PREFIX):
                    continue
                val = getattr(fn, attr)
                seen.add(val)
                attr = attr[len(SAVED_PREFIX):]
                if torch.is_tensor(val):
                    dot.edge(str(id(fn)), str(id(val)), dir="none")
                    dot.node(str(id(val)), get_var_name(val, attr), fillcolor='orange')
                if isinstance(val, tuple):
                    for i, t in enumerate(val):
                        if torch.is_tensor(t):
                            name = attr + '[%s]' % str(i)
                            dot.edge(str(id(fn)), str(id(t)), dir="none")
                            dot.node(str(id(t)), get_var_name(t, name), fillcolor='orange')


        # if hasattr(fn, 'variable'):
        #     # if grad_accumulator, add the node for `.variable`
        #     var = fn.variable
        #     seen.add(var)
        #     dot.node(str(id(var)), get_var_name(var), fillcolor='lightblue')
        #     dot.edge(str(id(var)), str(id(fn)))

        fn_name = get_fn_name(fn, show_attrs, max_attr_chars)

        # HACK START ==============>>
        attrs = {}
        _the_fn_name = str(type(fn).__name__)
        if "HistHackFnBackward" in _the_fn_name:
            # if hasattr(fn, "saved_tensors") and fn.saved_tensors:
            # saved tensor contains ord values of characters making up png path
            png = "".join(list(map(chr, list(fn.saved_tensors[0].numpy()))))
            attrs = {"image": png}
            fn_name = "" # dont display "PlotHackFnBackward" in the middle of histogram
        # <============== HACK END

        if hasattr(fn, "variable"):
            # add lightblue node for this AccumulateGrad function's parameter instead:
            var = fn.variable
            seen.add(var)
            dot.node(str(id(fn)), get_var_name(var), fillcolor='lightblue')
        else:
            # add the node for this grad_fn
            dot.node(str(id(fn)), fn_name, **attrs)

        # recurse
        if hasattr(fn, 'next_functions'):
            for u in fn.next_functions:
                if u[0] is not None:
                    dot.edge(str(id(u[0])), str(id(fn)))
                    add_nodes(u[0])

        # note: this used to show .saved_tensors in pytorch0.2, but stopped
        # working* as it was moved to ATen and Variable-Tensor merged
        # also note that this still works for custom autograd functions
        if hasattr(fn, 'saved_tensors'):
            for t in fn.saved_tensors:
                seen.add(t)
                dot.edge(str(id(t)), str(id(fn)), dir="none")
                dot.node(str(id(t)), get_var_name(t), fillcolor='orange')


    def add_base_tensor(var, color='darkolivegreen1'):
        if var in seen:
            return
        seen.add(var)
        dot.node(str(id(var)), get_var_name(var), fillcolor=color)
        if (var.grad_fn):
            add_nodes(var.grad_fn)
            dot.edge(str(id(var.grad_fn)), str(id(var)))
        if var._is_view():
            add_base_tensor(var._base, color='darkolivegreen3')
            dot.edge(str(id(var._base)), str(id(var)), style="dotted")


    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_base_tensor(v)
    else:
        add_base_tensor(var)

    resize_graph(dot)

    return dot

def make_dot_blitz(var, params=None, show_attrs=False, show_saved=False, max_attr_chars=50, graph_dir=None):
    """ Produces Graphviz representation of PyTorch Blitz model autograd graph.

    If a node represents a backward function, it is gray. Otherwise, the node
    represents a tensor and is either blue, orange, or green:
     - Blue: reachable leaf tensors that requires grad (tensors whose `.grad`
         fields will be populated during `.backward()`)
     - Orange: saved tensors of custom autograd functions as well as those
         saved by built-in backward nodes
     - Green: tensor passed in as outputs
     - Dark green: if any output is a view, we represent its base tensor with
         a dark green node.

    Args:
        var: output tensor
        params: dict of (name, tensor) to add names to node that requires grad
        show_attrs: whether to display non-tensor attributes of backward nodes
            (Requires PyTorch version >= 1.9)
        show_saved: whether to display saved tensor nodes that are not by custom
            autograd functions. Saved tensor nodes for custom functions, if
            present, are always displayed. (Requires PyTorch version >= 1.9)
        max_attr_chars: if show_attrs is `True`, sets max number of characters
            to display for any given attribute.
    """
    if LooseVersion(torch.__version__) < LooseVersion("1.9") and \
        (show_attrs or show_saved):
        warnings.warn(
            "make_dot: showing grad_fn attributes and saved variables"
            " requires PyTorch version >= 1.9. (This does NOT apply to"
            " saved tensors saved by custom autograd functions.)")

    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}
    else:
        param_map = {}

    node_attr = dict(
         style='filled',
         shape='box',
         align='left',
         fontsize='10',
         ranksep='0.1',
         height='0.2',
         fontname='monospace',
         image="",
     )
    dot = Digraph(node_attr=node_attr, graph_attr=dict(dpi="512.0"))# size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def get_var_name(var, name=None):
        if not name:
            name = param_map[id(var)] if id(var) in param_map else 'input'
        return '%s\n %s' % (name, size_to_str(var.size()))

    def find_attached_params_recursive(fn, found={}):
        assert not torch.is_tensor(fn)
        if hasattr(fn, "variable"):
            var = fn.variable
            found[get_var_name(var).split("\n")[0]] = var
        else:
            fn_name = get_fn_name(fn, show_attrs, max_attr_chars)
            # print(f"searching for rho belonging to a found mu, bwd: {fn_name}")
        # recurse
        if hasattr(fn, 'next_functions'):
            for u in fn.next_functions:
                if u[0] is not None:
                    find_attached_params_recursive(u[0], found)
        return found



    def add_nodes(fn, params_explored=False):
        assert not torch.is_tensor(fn)
        if fn in seen:
            return
        seen.add(fn)

        if show_saved:
            for attr in dir(fn):
                if not attr.startswith(SAVED_PREFIX):
                    continue
                val = getattr(fn, attr)
                seen.add(val)
                attr = attr[len(SAVED_PREFIX):]
                if torch.is_tensor(val):
                    dot.edge(str(id(fn)), str(id(val)), dir="none")
                    dot.node(str(id(val)), get_var_name(val, attr), fillcolor='orange')
                if isinstance(val, tuple):
                    for i, t in enumerate(val):
                        if torch.is_tensor(t):
                            name = attr + '[%s]' % str(i)
                            dot.edge(str(id(fn)), str(id(t)), dir="none")
                            dot.node(str(id(t)), get_var_name(t, name), fillcolor='orange')


        # if hasattr(fn, 'variable'):
        #     # if grad_accumulator, add the node for `.variable`
        #     var = fn.variable
        #     seen.add(var)
        #     dot.node(str(id(var)), get_var_name(var), fillcolor='lightblue')
        #     dot.edge(str(id(var)), str(id(fn)))

        fn_name = get_fn_name(fn, show_attrs, max_attr_chars)

        attrs = {}
        _the_fn_name = str(type(fn).__name__)
        is_histogram = False
        if "HistHackFnBackward" in _the_fn_name:
            # if hasattr(fn, "saved_tensors") and fn.saved_tensors:
            # saved tensor contains ord values of characters making up png path
            png = "".join(list(map(chr, list(fn.saved_tensors[0].numpy()))))
            attrs = {"image": png}
            fn_name = "" # dont display "PlotHackFnBackward" in the middle of histogram
            is_histogram = True

        dont_recurse = False
        if "AddBackward0"in _the_fn_name:
            # find parameters mu and rho in subtree:
            found_mu = False
            mu = None
            mu_parent = None
            found_rho = False
            rho = None
            rho_parent = None
            # search for mu directly above this function:
            if hasattr(fn, "next_functions"):
                for u in fn.next_functions:
                    u0 = u[0]
                    if u0 is not None and hasattr(u0, "variable"):
                        mu_name = get_var_name(u0.variable).split("\n")[0]
                        if mu_name.endswith("mu"):
                            found_mu = True
                            mu = u0.variable
                            mu_parent = mu_name.replace("mu", "")
                            break
            if found_mu:
                # look for params higher up in the tree to find rho
                # print("===")
                # print(f"found mu({mu_name}; shape={mu.shape})")
                found_params = find_attached_params_recursive(fn)
                # look for mu, rho
                for rho_name, param in found_params.items():
                    if rho_name.endswith("rho"):
                        rho_parent = rho_name.replace("rho", "")
                        if  (mu_parent==rho_parent):
                            found_rho = True
                            rho = param
                            # print(f"found rho({rho_name}; shape={rho.shape}) !")
                            break
                        else:
                            rho_parent = None
                if not found_rho:
                    # print(f"found mu, but no rho :(")
                    # print(found_params)
                    pass
            if found_mu and found_rho:
                # save histogram
                # # print(f"Found normal dist parameters:")
                # # print(f"mu({mu_name}): {mu.shape}")
                # # print(f"rho({rho_name}): {rho.shape}")
                avgd = []
                for t in mu, rho:
                    meaned_dims = len(t.shape) - 1
                    if meaned_dims >= 1:
                        meaned_t = t
                        for _ in range(meaned_dims):
                            meaned_t = meaned_t.mean(1)
                        t = meaned_t
                    assert len(t.shape) == 1, t.shape
                    avgd += [t]
                png = mu_parent + "png"
                save_ridge_gauss(*avgd, os.path.join(graph_dir, png))
                attrs = {"image":  png}
                fn_name = "" # dont display "AddBackward0" in the middle of histogram
                dont_recurse = True


        if hasattr(fn, "variable"):
            # add lightblue node for this AccumulateGrad function's parameter instead:
            var = fn.variable
            seen.add(var)
            dot.node(str(id(fn)), get_var_name(var), fillcolor='lightblue')
        else:
            # add the node for this grad_fn
            dot.node(str(id(fn)), fn_name, **attrs)


        # recurse
        if (not dont_recurse) and hasattr(fn, 'next_functions'):
            for u in fn.next_functions:
                if u[0] is not None:
                    dot.edge(str(id(u[0])), str(id(fn)))
                    add_nodes(u[0])

        # note: this used to show .saved_tensors in pytorch0.2, but stopped
        # working* as it was moved to ATen and Variable-Tensor merged
        # also note that this still works for custom autograd functions
        if (not is_histogram) and hasattr(fn, 'saved_tensors'):
            for t in fn.saved_tensors:
                seen.add(t)
                dot.edge(str(id(t)), str(id(fn)), dir="none")
                dot.node(str(id(t)), get_var_name(t), fillcolor='orange')


    def add_base_tensor(var, color='darkolivegreen1'):
        if var in seen:
            return
        seen.add(var)
        dot.node(str(id(var)), get_var_name(var), fillcolor=color)
        if (var.grad_fn):
            add_nodes(var.grad_fn)
            dot.edge(str(id(var.grad_fn)), str(id(var)))
        if var._is_view():
            add_base_tensor(var._base, color='darkolivegreen3')
            dot.edge(str(id(var._base)), str(id(var)), style="dotted")


    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_base_tensor(v)
    else:
        add_base_tensor(var)

    resize_graph(dot)

    return dot




def make_dot_from_trace(trace):
    """ This functionality is not available in pytorch core at
    https://pytorch.org/docs/stable/tensorboard.html
    """
    # from tensorboardX
    raise NotImplementedError("This function has been moved to pytorch core and "
                              "can be found here: https://pytorch.org/docs/stable/tensorboard.html")


def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.

    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)

def save_ridge_gauss(
        mu: Tensor = torch.randn(5),
        rho: Tensor = torch.rand(5),
        file_path: Optional[str] = None,
        N: int = 1000,
        min_pct: float = 0.20,
        max_pct: float = 0.30,
        first_n_dims: Optional[int] = 6,
    ):
    assert len(mu.shape) == 1
    assert mu.shape == rho.shape

    D = mu.shape[0] if (first_n_dims is None) else min(first_n_dims, mu.shape[0])

    scales = torch.log1p(torch.exp(rho))

    normals = []
    min_ppfs = []
    max_ppfs = []
    labels = []
    for d in range(D):
        loc = mu[d].item()
        scale = scales[d].item()
        Normal = norm(loc=loc,scale=scale)

        normals += [Normal]
        min_ppfs += [Normal.ppf(min_pct)]
        max_ppfs += [Normal.ppf(max_pct)]
        labels += [f"dim={d}; loc={round(loc, 4)}, scale={round(scale, 5)}"]

    # support is range from leftest left to rightest right
    support = np.linspace(min(min_ppfs), max(max_ppfs), num=N)
    # support = np.linspace(-5.,5., num=N)

    pdfs = np.hstack([dist.pdf(support) for dist in normals])
    g = np.repeat(np.array(labels), N)

    df = pd.DataFrame(dict(x=pdfs, g=g))

    # ridge plot;
    # from https://seaborn.pydata.org/examples/kde_ridgeplot.html

    # Initialize the FacetGrid object
    # pal = sns.cubehelix_palette(D, rot=-.25, light=.7)
    # ridge = sns.FacetGrid(data, aspect=15, height=.5, palette=pal)
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    ridge = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)

    # Draw the densities in a few steps
    ridge.map(
        sns.kdeplot,
        "x",
        bw_adjust=.5,
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.5
    )
    ridge.map(
        sns.kdeplot,
        "x",
        clip_on=False,
        color="w",
        lw=2,
        bw_adjust=.5
    )

    # passing color=None to refline() uses the hue mapping
    ridge.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)

    ridge.map(label, "x")

    # Set the subplots to overlap
    ridge.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    ridge.set_titles("")
    ridge.set(yticks=[], ylabel="")
    ridge.despine(bottom=True, left=True)

    if file_path is not None:
        plt.savefig(file_path)
    plt.gcf().clear()


