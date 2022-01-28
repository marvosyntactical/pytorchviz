from .dot import make_dot, make_dot_blitz
import graphviz

import torch
from torch import nn, Tensor

from typing import List, Dict, Any, Set, Type, Union, Tuple, Optional
from copy import deepcopy
import logging
import os
from glob import glob
import shutil
import subprocess
import matplotlib.pyplot as plt

# add module types which should not be recorded here
IGNORED_MODULE_TYPES: Set[Type] = {
    nn.Flatten,
    nn.Dropout,
}

__all__ = [
    "HistManager",
    "IGNORED_MODULE_TYPES"
]

class Histogram(nn.Module):
    """
    Histogram Creator to be inserted by HistManager
    before and/or after every module
    """
    def __init__(
            self,
            name: str,
            node_id: str,
            node_title: str,
            hist_indices: List[int],
            hist_freq: int,
            bins: int,
            hist_dir: str,
            latests_dir: str,
        ):
        """
        :param name: is name of model/experiment associated with HistogramManager
        :param node_id: is name of plot subdirectory
        :param node_title: is title of histogram
        :param hist_indices: list of indices i s.t. for the ith batch a histogram should be made
        :param hist_freq: frequency of every how many batches I should save histograms
        :param bins: number of x axis bins of histograms
        :param hist_dir: directory in which the histograms should be saved
        :param latests_dir: directory in which the latest histograms are saved
        """
        super().__init__()


        self.name = name
        self.node_id = node_id.replace(" ", "_")
        self.title = node_title

        self.hist_freq = hist_freq
        self.hist_indices = hist_indices
        self.bins = bins

        self.batch_counter = 0

        assert os.path.isdir(hist_dir), f"hist_dir {hist_dir} must be created by HistManager"

        self.point_hist_dir = os.path.join(hist_dir, self.node_id)
        if not os.path.isdir(self.point_hist_dir):
            os.mkdir(self.point_hist_dir)

        ext = "png"
        self.hist_tmpl = self.node_id + "_{}."+ext

        # will create copy of file when plotting:
        latest_copy = self.node_id+"_LATEST." + ext # this is overwritten always
        self.latest_path = os.path.join(latests_dir, latest_copy)

        # encoding of path of copy of latest png for saving in backward context
        self.encoded_latest_copy = torch.Tensor(list(map(ord, latest_copy)))
        self.encoded_latest_copy.requires_grad_(False)

        # will be set to true on successive forward passes (e.g. LSTM unroll)
        self.dont_create_hist = False

        class HistHackFn(torch.autograd.Function):
            """
            Hacky torch.autograd.Function used by HistManager
            to save the path of the latest saved image.
            For visualizing the activations with pytorchviz.
            """
            @staticmethod
            def forward(ctx, x: torch.Tensor):
                """
                HistHackFnFwd saves a tensor
                containing the path to a png created during the fwd pass
                for retrieval during torchviz's backward pass.
                """
                ctx.save_for_backward(self.encoded_latest_copy)
                return x

            @staticmethod
            def backward(ctx, grad_output):
                """ Identity """
                return grad_output

        self.hist_hack_fn = HistHackFn.apply


    def forward(self, *args, **kwargs) -> None:
        # assert False, (args, len(args)) #, [x.shape for x in args])
        # print(args[0][0])
        x = args[0][0]
        if isinstance(x, Tuple):
            assert len(x) == 1, x
            x = x[0]
        if not self.dont_create_hist:
            self.create_hist(x)
        x = self.hist_hack_fn(x)
        return x

    def create_hist(self, x: Tensor):

        count = self.batch_counter
        # sometimes store matplotlib histogram
        hist_because_idx = count in self.hist_indices
        freq = self.hist_freq
        hist_because_freq = False if freq <= 0 else count % freq == 0

        if hist_because_idx or hist_because_freq:

            # expensive, so make frequency low/dont create too many hists
            data = x.detach().reshape(-1).cpu().numpy() # expensive
            plt.hist(data, bins=self.bins) # expensive
            plt.ylabel("Frequency of bin")
            plt.title(f"{self.name}...{self.title}, batch #{count})", fontsize=16)
            plt.gcf()

            # save
            fig_path = os.path.join(
                self.point_hist_dir,
                self.hist_tmpl.format(count)
            )
            plt.savefig(fig_path)

            # update latest png (have to write separate png instead of symlink to above fig_path,
            # because graphviz considers symlinks unsafe)
            plt.savefig(self.latest_path)
            plt.gcf().clear()

            self.batch_counter += 1



class HistManager:
    """
    Manages histogram creation/saving for a given pytorch model by inserting
    forward pre hooks and (post) hooks in modules recursively.
    (see PyTorch nn.Module register*hook API).
    Saves hists in a given folder;
    to be inserted into a model graph by make_histgram_plot.
    """
    def __init__(
            self,
            name: str,
            hist_freq: int = 100,
            hist_indices: List[int] = [],
            bins: int = 1000,
            hist_superdir: str = "plots",
            record_input_dists: bool = True,
            record_output_dists: bool = True,
            banned_types: Set[Type] = IGNORED_MODULE_TYPES,
            verbose: bool = False,
            blitz: bool = False,
        ):
        """
        :param name: model name; also directory name to cache hists under
        :param hist_freq: frequency of every how many batches I should save histograms
        :param hist_indices: list of indices i s.t. for the ith batch a histogram should be made
        :param bins: number of x axis bins of histograms
        :param hist_superdir: (existing) directory in which the directory "name" is created
        :param record_input_dists: whether to add a histogram for each module's input
        :param record_output_dists: whether to add a histogram for each module's output
        :param banned_types: list of nn.module subtypes which should be ignored, e.g. flatten
        :param verbose: NotImplemented
        :param blitz: blitz compatibility
        :return:
        """
        assert record_input_dists or record_output_dists, f"record at least one type of dist"
        self.record_input_dists = record_input_dists
        self.record_output_dists = record_output_dists

        self.banned_types = banned_types
        self.dotify = make_dot if not blitz else make_dot_blitz

        ext = "png" # saves .png images

        self.bins = bins

        self.name = name.replace(" ","_")
        self.hist_freq = hist_freq
        self.hist_indices = hist_indices

        if not os.path.exists(hist_superdir):
            os.mkdir(hist_superdir)

        name_dir = os.path.join(hist_superdir, self.name)

        if not os.path.exists(name_dir):
            os.mkdir(name_dir)

        self.model_graph_dir = os.path.join(name_dir, "model_graphs")

        if not os.path.exists(self.model_graph_dir):
            os.mkdir(self.model_graph_dir)

        self.hist_dir = os.path.join(name_dir, "hists")

        if not os.path.exists(self.hist_dir):
            os.mkdir(self.hist_dir)

        if verbose:
            raise NotImplementedError(f"TODO implement logger")
        self.logger = logging.getLogger(name=self.name)

        # FOR MODEL GRAPH VISUALIZATION:
        self.latests_dir = latests_dir = os.path.join(self.hist_dir, "LATEST_HISTOGRAMS")
        if not os.path.exists(latests_dir):
            os.mkdir(latests_dir)

    def process_model(self, model: nn.Module, inplace: bool = True, model_name: str="model") -> nn.Module:
        # insert Histograms throughout model (inplace), recursively
        if not inplace:
            model = deepcopy(model)

        # let n be the number of modules in the model.
        # these lists have length n
        self.hist_points: List[Dict[str, Histogram]] = []
        self.hook_handles: List[Dict[str, Any]] = [] # contains handles s.t. calling handle.remove() removes the Hist Point
        self.module_types: List[type] = []
        self.module_names: List[str] = []

        def prep_module(
            module: nn.Module,
            module_number: int = 0,
            parent_number: int = 0,
            name: str = "model"
            ) -> (nn.Module, Dict):

            child_module_number = module_number + 1 # increment DFS counter by 1 for current module before diving down further
            for name, child in module.named_children():
                # ===== Depth First Search down module graph ========
                _, child_module_number = prep_module(
                    child,
                    child_module_number, # numbering is DFS preorder
                    module_number, # this module is parent of children in loop
                    name
                )

            # naming style: layer_name(layer_type)
            module_type = type(module)
            module_name = f"""{name}({str(module_type).split(".")[-1].replace("'>", "")})"""
            node_title = module_name
            module_id = name + f"_{module_number}"

            self.module_types += [module_type]
            self.module_names += [module_name]
            self.hist_points += [{}]
            self.hook_handles += [{}]

            if module_type not in self.banned_types:

                if self.record_input_dists:
                    # init input hist point:
                    HPinput = Histogram(
                        name=self.name,
                        node_id=module_id + " input",
                        node_title=module_name + " input",
                        hist_indices=self.hist_indices,
                        hist_freq=self.hist_freq,
                        bins=self.bins,
                        hist_dir=self.hist_dir,
                        latests_dir=self.latests_dir,
                    )
                    hook = lambda mod, x: HPinput(x)
                    # add pre hook + handle:
                    pre_hook_handle = module.register_forward_pre_hook(hook)

                    self.hist_points[-1]["input"] = HPinput
                    self.hook_handles[-1]["input"] = pre_hook_handle


                if self.record_output_dists:
                    # init output hist point:
                    HPoutput = Histogram(
                        name=self.name,
                        node_id=module_id + "_output",
                        node_title=module_name + " output",
                        hist_indices=self.hist_indices,
                        hist_freq=self.hist_freq,
                        bins=self.bins,
                        hist_dir=self.hist_dir,
                        latests_dir=self.latests_dir,
                    )

                    hook = lambda mod, x, y: HPoutput(((y,),))

                    # add pre hook + handle:
                    # add post hook + handle:
                    post_hook_handle = module.register_forward_hook(hook)

                    self.hist_points[-1]["output"] = HPoutput
                    self.hook_handles[-1]["output"] = post_hook_handle

            return module, child_module_number

        model, _ = prep_module(model, name=model_name)
        self.model = model
        return model

    def stop(self) -> nn.Module:
        # call .remove() on all registered hook handles
        # do not call if you still want to plot model graphs
        for i, handles in enumerate(self.hook_handles):
            if handles.get("input", None) is not None:
                handles["input"].remove()

            if handles.get("output", None) is not None:
                handles["output"].remove()

    def plot_model_graph(
            self,
            input_data: Union[Tuple[Tensor],Tensor], # B x T x D
            graph_name: Optional[str] = None,
            format="png",
        ):
        # hists do not represent activations for input data; input data is only used by
        # torchviz for graph creation; hists are saved by HistManager
        # make sure you are not within torch.no_grad() context!

        if graph_name is None:
            graph_name = self.name

        out_dir = self.model_graph_dir
        img_dir = self.latests_dir

        model = self.model

        images = glob(os.path.join(img_dir, "*"))
        for img in images:
            shutil.copy(img, out_dir)

        # only one time step for lstm!
        if "lstm" in str(type(model)).lower():
            assert len(input_data.shape) == 3, input_data.shape
            input_data = input_data[:,0].unsqueeze_(1)

        # forward pass with gradient creation
        input_data.requires_grad_(True)
        out = model(input_data)

        d = self.dotify(
            out.mean(),
            params=dict(model.named_parameters()),
            graph_dir=out_dir,
            show_saved=False,
        )
        # g = str(d)
        # G = graphviz.Source(g)

        # all images in graph must be in directory of out_f.
        graph_file = os.path.join(out_dir, graph_name)
        d.render(graph_file, format="png")

        # graphviz creates two files, "graph_file" <- wrong; and "graph_file.png" <- correct
        os.remove(graph_file)

        # remove the copied images after they were used to render the graph
        for img in images:
            im = img.split("/")[-1]
            os.remove(os.path.join(out_dir, im))


        self.logger.info(f"\nSaved model graph and corresponding images in {out_dir}!")
        return


