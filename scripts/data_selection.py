from __future__ import print_function
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader

import os
import glob
from core.trak import TRAKer
from core.trak.utils import get_matrix_mult
import seaborn as sns
import torch
import pickle
from multiprocessing import Pool
import wandb


import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.score import SegmentationMetric
from core.utils.visualize import get_color_pallete
from core.utils.logger import setup_logger
from core.utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler
from core.utils.loss import get_segmentation_loss

from train import parse_args


def numba_topk_2d_axis0(args):
    arr, k = args

    # Find the partitioned indices for top-k elements along axis 0
    partitioned_indices = np.argpartition(arr, -k, axis=0)[-k:]

    # Retrieve the k largest values based on the partitioned indices
    k_largest_values = np.take_along_axis(arr, partitioned_indices, axis=0)

    # Sort the k largest values along axis 0 to get sorted indices within top-k
    sorted_indices_within_k = np.argsort(-k_largest_values, axis=0)

    # Get the sorted indices for the original array
    sorted_indices = np.take_along_axis(partitioned_indices, sorted_indices_within_k, axis=0)

    # Get the sorted scores using the sorted indices
    sorted_scores = np.take_along_axis(arr, sorted_indices, axis=0)

    return sorted_scores, sorted_indices


def multi_process_sort(array, k=20000):
    process_num = os.cpu_count()
    data_splits = np.array_split(array, process_num, axis=1)

    # Prepare arguments as tuples
    args = [(split, k) for split in data_splits]

    with Pool(processes=process_num) as pool:
        # Map function that takes a tuple of arguments
        results = pool.map(numba_topk_2d_axis0, args)

    # Separate the sorted indices and scores from the results
    sorted_indices = np.concatenate([result[1] for result in results], axis=1)
    sorted_scores = np.concatenate([result[0] for result in results], axis=1)

    return sorted_scores, sorted_indices


def get_score(args):

    def save_selected_index(selected_train_files, weight):
        train_data_paths = cfg.train_data_path
        train_files = train_set.data_loaded_keys

        min_val, max_val = np.min(weight), np.max(weight)
        scaled = 1 + 2 * (weight - min_val) / (max_val - min_val)
        weight = np.round(scaled).astype(int)

        # selected_dataset = []
        for train_data_path in train_data_paths:
            phase, dataset_name = train_data_path.split('/')[-2], train_data_path.split('/')[-1]
            cache_path = os.path.join(cfg['cache_path'], dataset_name, phase)
            file_list_path = os.path.join(cache_path, 'file_list.pkl')
            with open(file_list_path, 'rb') as f:
                file_list = pickle.load(f)
            for k, v in file_list.items():
                file_list[k]['gradient_score'] = 0

            for idx, k in enumerate(selected_train_files):
                file_ = train_files[k]
                if file_list.get(file_) is not None:
                    file_list[file_]['gradient_score'] = weight[idx]
                    # file_path = file_list[file_]['h5_path']
                    # with h5py.File(file_path, 'r') as f:
                    #     group = f[file_]
                    #     record = {k: group[k][()].decode('utf-8') if group[k].dtype.type == np.string_ else group[k][()] for
                    #               k in group.keys()}
                    # selected_dataset.append(record['location'])
            # save file_list
            with open(file_list_path, 'wb') as f:
                pickle.dump(file_list, f)
        print('data selection saved')

    def transform_batch(batch):
        inp_dict = batch['input_dict']
        batch = [x.to(device) for x in inp_dict.values() if type(x) == torch.Tensor]
        keys = [k for k in inp_dict.keys() if type(inp_dict[k]) == torch.Tensor]
        return keys, batch

    def visualize_score(similarity_matrix):

        # 限制数据的最大最小值为正负500
        # similarity_matrix = np.clip(similarity_matrix, -500, 500)

        # 计算平均相似度、最大相似度和最小相似度
        mean_similarities = np.mean(similarity_matrix, axis=1)
        max_similarities = np.max(similarity_matrix, axis=1)
        min_similarities = np.min(similarity_matrix, axis=1)

        # 创建一个图形
        plt.figure(figsize=(18, 6))

        cmap = sns.color_palette("colorblind", 3)

        # 平均相似度分布
        plt.subplot(1, 3, 1)
        sns.histplot(mean_similarities, kde=True, bins=20, color=cmap[0])
        plt.title('Average Score Distribution')
        plt.xlabel('Average Score')
        plt.ylabel('Frequency')

        # 最大相似度分布
        plt.subplot(1, 3, 2)
        sns.histplot(max_similarities, kde=True, bins=20, color='red')
        plt.title('Maximum Score Distribution')
        plt.xlabel('Maximum Score')
        plt.ylabel('Frequency')

        # 最小相似度分布
        plt.subplot(1, 3, 3)
        sns.histplot(min_similarities, kde=True, bins=20, color='green')
        plt.title('Minimum Score Distribution')
        plt.xlabel('Minimum Score')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()
        return wandb.Image(plt)

    def data_selector(score, num_selected, feature_mat=None, lambda_param=None, method='random'):

        def cosine_L2(x, y):
            # return geomloss.utils.squared_distances(x, y)
            x = x.to('cuda')
            y = y.to('cuda')
            sim = get_matrix_mult(x[0], y[0])
            sim.mul_(-2)
            sim.add_(2)
            sim.abs_()
            sim.sqrt_()
            # print(torch.max(sim),torch.min(sim))
            sim = sim.to('cuda')
            return sim.unsqueeze(0)
        import geomloss
        distance = geomloss.SamplesLoss(
            loss='sinkhorn',
            cost=cosine_L2,
            debias=False,
            blur=0.01,
            # reach=0.5,
            potentials=False,
            backend='tensorized',
            scaling=0.5
        )
        assert num_selected <= score.shape[0]
        if method == 'random':
            selected_train_files = np.random.choice(score.shape[0], num_selected, replace=False)
        elif method == 'dsdm':
            avg_score = np.mean(score, axis=1)
            closest_training_ids = np.argsort(avg_score)
            selected_train_files = closest_training_ids[-num_selected:].reshape(-1)
        elif method == 'ot_optimal':

            candidate_number, target_number = score.shape
            # Sort score once
            # score = np.abs(score)  # Ensure score is absolute
            k_neigh = 100
            sorted_score, sorted_indices = multi_process_sort(score, k=k_neigh)

            g = traker.g_normed.to(torch.float32).to('cpu')
            g_target = traker.g_target.to(torch.float32).to('cpu')
            weight = 0
            from sklearn.model_selection import KFold

            # Split into 5 folds using KFold
            kf = KFold(n_splits=20, shuffle=True, random_state=42)

            selected_train_files = []
            # Perform 5-fold cross-validation
            for fold, (train_index, test_index) in enumerate(kf.split(np.arange(target_number))):
                print(f"Fold {fold + 1}")
                target_val = test_index
                target_test = train_index

                g_target_fold = g_target[target_test]

                min_c_fold = 10000
                dist_threshold = 0.0

                for k in range(1, k_neigh):
                    # Use the pre-sorted indices and scores
                    sorted_indices_k = sorted_indices[:k, target_val].reshape(-1)
                    sorted_score_k = sorted_score[:k, target_val].reshape(-1)
                    score_above_threshold = sorted_score_k > dist_threshold

                    # Filter indices above threshold
                    sorted_indices_k = sorted_indices_k[score_above_threshold]

                    # Get unique indices
                    unique_indices, cnt = np.unique(sorted_indices_k, return_counts=True)
                    print(f'k: {k}, unique_indices: {len(unique_indices)}')
                    # Calculate distance for this fold
                    c = distance(g[unique_indices].to('cuda'), g_target_fold.to('cuda'))
                    print(f'k: {k}, distance: {c.item()}')
                    if c.item() < min_c_fold:
                        min_c_fold = c.item()
                    else:
                        break

                print(f'Final distance for fold {fold + 1}: {min_c_fold}')
                selected_train_files.append(unique_indices)
                print(f'Selection ratio for fold {fold + 1}: {len(unique_indices) / candidate_number}')

            # Print overall result
            selected_train_files = np.concatenate(selected_train_files)
            unique_indices, cnt = np.unique(selected_train_files, return_counts=True)
            selected_train_files = unique_indices

            distance.potentials = True
            # f, _ = distance(g[unique_indices].to('cuda'), g_target.to('cuda'))
            # weight = f[0].detach().cpu().numpy()
            weight = np.ones(candidate_number)

            # weight[:]=-1
            # weight[unique_indices] = seleted_weight
            # weight = 0
            print(f'Overall selection ratio: {len(unique_indices) / candidate_number}')

        elif method == 'fixed_size':
            candidate_number, target_number = score.shape

            g = traker.g_normed.to(torch.float32).to('cpu')
            g_target = traker.g_target.to(torch.float32).to('cpu')
            # score = np.abs(score)
            sorted_score, sorted_indices = multi_process_sort(score, k=100)
            unique_set = set()
            for i in range(sorted_indices.shape[0]):
                this_layer = np.unique(sorted_indices[i])
                updated_set = unique_set.union(this_layer)
                # count the number of unique elements
                if len(updated_set) <= num_selected:
                    unique_set = updated_set
                else:
                    number_to_add = num_selected - len(unique_set)
                    diff = np.setdiff1d(this_layer, list(unique_set))

                    distance.potentials = True
                    f, _ = distance(g[diff].to('cuda'), g_target.to('cuda'))
                    seleted_weight = f[0].detach().cpu().numpy()
                    sorted_diff = np.argsort(seleted_weight)
                    unique_set = unique_set.union(diff[sorted_diff[number_to_add:]])
                    break
            distance.potentials = True
            g_selected = g[list(unique_set)]
            f, _ = distance(g_selected.to('cuda'), g_target.to('cuda'))
            weight = f[0].detach().cpu().numpy()
            selected_train_files = np.array(list(unique_set))
        elif method == 'greedy':
            sorted_score, sorted_indices = multi_process_sort(score)
            selected_index = []
            cnt = 0
            selected_feature = torch.zeros([num_selected, feature_mat.shape[1]], dtype=torch.float16).to(device)

            # use greedy method to select the top num_selected
            selected_index = []
            candidate_set = unique_indices
            current_distance = 10000
            for i in tqdm(range(sorted_indices.shape[0])):
                candidate_score = []
                candidate_index = []
                for index in candidate_set:
                    # calculate ot distance
                    merged_index = np.concatenate([selected_index, [index]])
                    g_selected = g[merged_index]
                    with torch.no_grad():
                        c = distance(g_selected, g_target)
                    candidate_score.append(c.item())
                    candidate_index.append(index)
                # get the index with the minimum distance
                top_candidate_index = np.argsort(candidate_score)[:100]
                min_distance = candidate_score[top_candidate_index]
                min_index = candidate_index[min_score_index]

                if min_distance >= current_distance:
                    break
                selected_index.append(min_index)
                candidate_set = np.setdiff1d(candidate_set, selected_index)
                if i % 100 == 0:
                    print(f'iteration {i}, distance: {min_distance}')

        print(f'len(selected_index): {len(selected_train_files)}, {len(np.unique(selected_train_files))} are unique')
        return selected_train_files, weight

    def tsne_visulization():
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        sample_num_train = 10000
        sample_num_val = 2000

        potential = weight
        print(1 / weight.shape[0])
        print(np.mean(potential), np.max(potential), np.min(potential))

        g = np.array(traker.g_normed.cpu())
        g_target = np.array(traker.g_target.cpu())
        g_random_sample = np.random.choice(len(g), min(len(g), sample_num_train), replace=False)
        g = g[g_random_sample]
        potential = potential[g_random_sample]
        g_target_random_sample = np.random.choice(len(g_target), min(len(g_target), sample_num_val), replace=False)
        g_target = g_target[g_target_random_sample]

        # # Initialize normalized array
        # lower_percentile, upper_percentile = 1, 99
        # lower_clip = np.percentile(potential, lower_percentile)
        # upper_clip = np.percentile(potential, upper_percentile)
        # # Clip values based on percentiles
        # potential = np.clip(potential, lower_clip, upper_clip)
        # # Separate positive and negative values
        # positive_mask = potential > 0
        # negative_mask = potential < 0
        # # Initialize normalized array
        # normalized_potential = np.zeros_like(potential)
        # # Normalize positive values to the range [0, 1]
        # max_positive = np.max(potential[positive_mask])
        # normalized_potential[positive_mask] = potential[positive_mask] / max_positive
        # # Normalize negative values to the range [0, -1]
        # min_negative = np.min(potential[negative_mask])
        # normalized_potential[negative_mask] = potential[negative_mask] / (-min_negative)

        all_gradient = np.concatenate([g, g_target], axis=0)

        print('start to fit')
        tsne = TSNE(n_jobs=16, perplexity=50, early_exaggeration=20)
        tsne_points = tsne.fit_transform(all_gradient)
        print('fit done')
        train_tsne = tsne_points[:len(g)]
        val_tsne = tsne_points[len(g):]
        # Define colorblind-friendly palette for validation points
        colorblind_palette = sns.color_palette("colorblind", as_cmap=False)

        # Normalize the potential values for gradient color mapping with a center at 0
        norm = TwoSlopeNorm(vmin=min(potential), vcenter=np.mean(potential), vmax=max(potential))
        cmap = plt.cm.get_cmap("coolwarm")  # Gradient from red (negative) to blue (positive)

        size = 8
        # Create the plot
        plt.figure(figsize=(10, 8))
        # Plot train points with gradient color based on potential
        scatter_train = plt.scatter(train_tsne[:, 0], train_tsne[:, 1],
                                    c=potential, cmap=cmap, norm=norm,
                                    label='Candidate Points', s=size, alpha=1)
        # Plot validation points in a different solid color (colorblind-friendly)
        plt.scatter(val_tsne[:, 0], val_tsne[:, 1],
                    color=colorblind_palette[2], label='Target Points', s=size, alpha=1)

        # Add a colorbar for the potential gradient (train points)
        cbar = plt.colorbar(scatter_train, label='Potential (Importance)')
        # Add labels and title
        plt.xlabel('t-SNE Dim 1')
        plt.ylabel('t-SNE Dim 2')
        plt.title('t-SNE Visualization of Train and Validation Points')
        # Add a legend to distinguish train and validation points
        plt.legend()
        # Display the plot
        # plt.show()

        return wandb.Image(plt)

    def visualize_scale(scale):
        plt.figure(figsize=(6, 6))

        # plt.figure(figsize=(6, 6))
        # plt.hist(scale, bins=100, density=True, color='blue', alpha=0.7)
        # plt.title('Scale Distribution')
        # plt.xlabel('Value')
        # plt.ylabel('Density')

        scale = np.sort(scale)
        cdf = np.arange(1, len(scale) + 1) / len(scale)

        # Plot the CDF using matplotlib
        plt.figure(figsize=(6, 6))
        plt.plot(scale, cdf, color='blue', marker='.', linestyle='none')
        plt.title('Cumulative Distribution Function (CDF)')
        plt.xlabel('Value')
        plt.ylabel('Cumulative Probability')

        return wandb.Image(plt)

    def visualize_mean_trak():
        sorted_index = np.argsort(sim_to_center)
        import matplotlib.font_manager as fm

        font_properties = fm.FontProperties(family='Arial', weight='bold', size=10)

        vis_num = 4
        image_width = 6
        top_index = sorted_index[-vis_num:]
        bottom_index = sorted_index[:vis_num]
        # all_index = np.concatenate([top_index, bottom_index])
        # all_index = bottom_index
        image_list = []
        plt.figure(figsize=(vis_num * image_width, image_width))
        # Plot the bottom 10 images
        for plt_idx, i in enumerate(bottom_index):
            plt.subplot(1, vis_num, plt_idx + 1)
            vis_range = 60
            plt.xlim(-vis_range + 30, vis_range + 30)
            plt.ylim(-vis_range, vis_range)
            score_for_this = sim_to_center[i]
            _ = check_loaded_data(plt, train_loader.dataset[i], 0)
            # plot score under this subplot
            plt.text(0.5, -0.1, f'{score_for_this:.2f}', ha='center', color='black', fontproperties=font_properties,
                     transform=plt.gca().transAxes)
            plt.axis('off')  # Remove axes for a cleaner look
        image_list.append(wandb.Image(plt))

        # Plot the top 10 images
        plt.figure(figsize=(vis_num * image_width, image_width))
        for plt_idx, i in enumerate(top_index):
            plt.subplot(1, vis_num, plt_idx + 1)
            vis_range = 60
            plt.xlim(-vis_range + 30, vis_range + 30)
            plt.ylim(-vis_range, vis_range)
            score_for_this = sim_to_center[i]
            _ = check_loaded_data(plt, train_loader.dataset[i], 0)
            # plot score under this subplot
            plt.text(0.5, -0.1, f'{score_for_this:.2f}', ha='center', color='black', fontproperties=font_properties,
                     transform=plt.gca().transAxes)
            plt.axis('off')  # Remove axes for a cleaner look
        image_list.append(wandb.Image(plt))

        return image_list

    def visualize_supportive_and_negative():
        total_num = len(val_loader.dataset)
        sample_interval = max(total_num // 30, 1)
        image_list = []
        for i in range(0, total_num, sample_interval):
            fig, ax = plt.subplots(1, 5, figsize=(20, 4))

            val_data = val_loader.dataset[i]
            val_path = val_data[-1]
            city = val_path.split('_')[0]
            val_image_path = os.path.join(val_dataset.root,'leftImg8bit','val',city,val_path)
            val_image = plt.imread(val_image_path)
            ax[0].imshow(val_image)

            # Get top and bottom TRAK scorers
            top_scores = _scores[:, i].argsort()[-2:][::-1]
            bottom_scores = _scores[:, i].argsort()[:2][::-1]

            # Plot top TRAK scorers
            for ii, train_im_ind in enumerate(top_scores):
                train_data = train_loader.dataset[train_im_ind]
                train_path = train_data[-1]
                gta_root = train_dataset.root.replace('citys', 'gta5')
                train_image_path = os.path.join(gta_root, 'images', train_path)
                train_image = plt.imread(train_image_path)
                ax[ii + 1].imshow(train_image)

            # Plot bottom TRAK scorers
            for ii, train_im_ind in enumerate(bottom_scores):
                train_data = train_loader.dataset[train_im_ind]
                train_path = train_data[-1]
                gta_root = train_dataset.root.replace('citys', 'gta5')
                train_image_path = os.path.join(gta_root, 'images', train_path)
                train_image = plt.imread(train_image_path)
                ax[ii + 3].imshow(train_image)

            plt.tight_layout()
            # plt.show()
            image_list.append(wandb.Image(plt))
        return image_list

    device = torch.device(args.device)

    save_dir = './trak_results'

    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    train_dataset = get_segmentation_dataset(args.dataset, split='train', mode='train', transform=input_transform)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=args.batch_size,
                                   num_workers=args.workers,
                                   pin_memory=True,
                                   drop_last=False)
    # dataset and dataloader
    val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='testval', transform=input_transform)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 num_workers=args.workers,
                                 batch_size=args.batch_size,
                                 pin_memory=True)

    # create network
    BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
    model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                   aux=args.aux, pretrained=True, pretrained_base=False,
                                   local_rank=args.local_rank,
                                   norm_layer=BatchNorm2d).to(device)
    criterion = get_segmentation_loss(args.model, use_ohem=args.use_ohem, aux=args.aux,
                                      aux_weight=args.aux_weight, ignore_index=-1).to(device)

    model.to(device)
    model.criterion = criterion

    ckpt_dir = os.path.expanduser(args.save_dir)
    ckpts = glob.glob(os.path.join(ckpt_dir, '*.pth'))
    ckpt_files = sorted(ckpts, key=os.path.getmtime)
    ckpts = ckpt_files[:-1]
    print('use ckpts:', ckpts)
    ckpts = [torch.load(ckpt, map_location=device) for ckpt in ckpts]

    exp_name = args.save_dir.split('/')[-1]

    traker = TRAKer(model=model,
                    task='semantic_segmentation',
                    proj_dim=4096,
                    train_set_size=len(train_dataset), device=device, load_from_save_dir=True,
                    save_dir=save_dir,
                    use_half_precision=False)

    for model_id, ckpt in enumerate(tqdm(ckpts)):
        traker.load_checkpoint(ckpt, model_id=model_id)
        for batch in tqdm(train_loader):
            traker.featurize(batch=batch, num_samples=batch[0].shape[0])
    traker.finalize_features()

    for model_id, ckpt in enumerate(tqdm(ckpts)):
        traker.start_scoring_checkpoint(exp_name=exp_name,
                                        checkpoint=ckpt,
                                        model_id=model_id,
                                        num_targets=len(val_dataset))
        for batch in val_loader:
            traker.score(batch=batch, num_samples=batch[0].shape[0])

    #_scores = traker.finalize_scores(exp_name=exp_name)
    _scores = np.random.rand(len(train_dataset), len(val_dataset))

    num_select = int(0.2 * len(train_dataset))
    select_method = 'fixed_size'
    print(_scores[:10,:10])
    selected_file, weight = data_selector(_scores, num_select, feature_mat=0, lambda_param=0.5, method=select_method)
    selected_file = np.random.choice(len(train_dataset), num_select, replace=False)
    weight = np.random.rand(num_select)
    #save_selected_index(selected_file, weight)

    wandb_image = {}
    #wandb_image['tsne'] = tsne_visulization()
    #wandb_image['score'] =visualize_score(_scores)
    wandb_image['attribution'] = visualize_supportive_and_negative()
    #wandb_image['mean_trak'] = visualize_mean_trak()
    #wandb_image['scale'] = visualize_scale(weight)

    wandb.init(project='unitraj', name=exp_name)
    wandb.log(wandb_image)





if __name__ == "__main__":

    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    # TODO: optim code
    args.save_pred = True
    if args.save_pred:
        outdir = '../runs/pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(),
                          filename='{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset), mode='a+')

    get_score(args)


