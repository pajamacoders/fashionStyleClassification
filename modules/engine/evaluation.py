import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from numbers import Number
from modules.utils.metric import APScorer, AverageMeter
from sklearn.metrics import classification_report 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def do_eval(
	model,
	query_loader,
	candidate_loader,
	gt,
	lt,
	attrs,
	device,
	logger,
	epoch=-1,
	beta=0.6
):
	mAPs = AverageMeter()

	logger.info("Begin evaluation.")
	model.eval()

	logger.info("Forwarding query images...")
	q_feats, q_values, q_preds_label = extract_features(model, query_loader, gt, lt, device, len(attrs), beta=beta)
	logger.info("Forwarding candidate images...")
	c_feats, c_values, c_preds_label = extract_features(model, candidate_loader, gt, lt, device, len(attrs), beta=beta)
	q_preds_label+=c_preds_label
	total_values = np.concatenate([q_values[0], c_values[0]])
	for i, attr in enumerate(attrs):
		mAP = mean_average_precision(q_feats[i], c_feats[i], q_values[i], c_values[i])
		logger.info(f"{attr} MeanAP: {100.*mAP:.4f}")
		mAPs.update(mAP, q_feats[i].shape[0])

	logger.info(f"Total MeanAP: {100.*mAPs.avg:.4f}")
	compute_matrics(q_preds_label, total_values, logger, epoch, mAPs.avg)
	return mAPs.avg

def compute_matrics(pred, targets, logger, epoch, mAPs):
    pred_values = np.concatenate(pred)
    eval_results = {}
    topk_= (1,)  
    clsname = ['formal', 'semi-formal', 'casual']#['mode', 'feminine', 'girlish', 'kireime-casual', 'conservative', 'retro', 'ethnic', 'street', 'dressy', 'rock', 'natural']
    acc = accuracy_numpy(pred_values, targets, topk=topk_)
    for key, val in zip(topk_,acc):
        eval_results[f'top_{key}']=val
    pred_targets = pred_values.argmax(axis=-1)
    cls_report = classification_report(targets.tolist(),pred_targets.tolist(), target_names=clsname, output_dict=True )
    for k,v in cls_report.items():
        eval_results[k]=str(v)
        
    met=f'epoch:{epoch}- mAPs:{mAPs:.4f}, '
    with open(f'kfashion_classification/test_result_20220112.txt', 'a') as f:
            f.write(met+'\n')
            for k,v in eval_results.items():
                f.write(f'{k}: {v}\n')

            f.write('\n\n')

def extract_features(model, data_loader, gt, lt, device, n_attrs, beta=0.6):
	feats = []
	indices = [[] for _ in range(n_attrs)]
	values = []
	pred_values =[]
	with tqdm(total=len(data_loader)) as bar:
		cnt = 0
		for idx, batch in enumerate(data_loader):
			x, a, v = batch
			a = a.to(device)

			out, cat_vals = process_batch(model, x, a, gt, lt, device, beta=beta)
			feats.append(out.cpu().numpy())
			values.append(v.numpy())
			pred_values.append(cat_vals.cpu().numpy())
			for i in range(a.size(0)):
				indices[a[i].cpu().item()].append(cnt)
				cnt += 1

			bar.update(1)

	
	feats = np.concatenate(feats)
	values = np.concatenate(values)
	
	feats = [feats[indices[i]] for i in range(n_attrs)]
	values = [values[indices[i]] for i in range(n_attrs)]

	return feats, values, pred_values


def process_batch(model, x, a, gt, lt, device, beta=0.6):
	gx = torch.stack([gt(i) for i in x], dim=0)
	gx = gx.to(device)
	with torch.no_grad():
		g_feats, attmap, cls_score = model(gx, a, level='global')

	if lt is None:
		return nn.functional.normalize(g_feats, p=2, dim=1), cls_score.softmax(dim=1)

	attmap = attmap.cpu().numpy()
	lx = torch.stack([lt(i, mask) for i, mask in zip(x, attmap)], dim=0)
	lx = lx.to(device)
	with torch.no_grad():
		l_feats, _ = model(lx, a, level='local')
	
	out = torch.cat((torch.sqrt(torch.tensor(beta)) * nn.functional.normalize(g_feats, p=2, dim=1),
			torch.sqrt(torch.tensor(1-beta)) * nn.functional.normalize(l_feats, p=2, dim=1)), dim=1)

	return out


def mean_average_precision(queries, candidates, q_values, c_values):
    '''
    calculate mAP of a conditional set. Samples in candidate and query set are of the same condition.
        cand_set: 
            type:   nparray
            shape:  c x feature dimension
        queries:
            type:   nparray
            shape:  q x feature dimension
        c_gdtruth:
            type:   nparray
            shape:  c
        q_gdtruth:
            type:   nparray
            shape:  q
    '''
 
    scorer = APScorer(candidates.shape[0])

    # similarity matrix
    simmat = np.matmul(queries, candidates.T)

    ap_sum = 0
    for q in range(simmat.shape[0]):
        sim = simmat[q]
        index = np.argsort(sim)[::-1]
        sorted_labels = []
        for i in range(index.shape[0]):
            if c_values[index[i]] == q_values[q]:
                sorted_labels.append(1)
            else:
                sorted_labels.append(0)
        
        ap = scorer.score(sorted_labels)
        ap_sum += ap

    mAP = ap_sum / simmat.shape[0]

    return mAP

def accuracy_numpy(pred, target, topk=1, thrs=0.):
    if isinstance(thrs, Number):
        thrs = (thrs, )
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.shape[0]
    pred_label = pred.argsort(axis=1)[:, -maxk:][:, ::-1]
    pred_score = np.sort(pred, axis=1)[:, -maxk:][:, ::-1]

    for k in topk:
        correct_k = pred_label[:, :k] == target.reshape(-1, 1)
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct_k = correct_k & (pred_score[:, :k] > thr)
            _correct_k = np.logical_or.reduce(_correct_k, axis=1)
            res_thr.append(_correct_k.sum() * 100. / num)
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def precision_recall_f1(pred, target, average_mode='macro', thrs=0.):
    """Calculate precision, recall and f1 score according to the prediction and
    target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        tuple: tuple containing precision, recall, f1 score.

            The type of precision, recall, f1 score is one of the following:

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """

    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')

    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    assert (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)),\
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    if isinstance(thrs, Number):
        thrs = (thrs, )
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    label = np.indices(pred.shape)[1]
    pred_label = np.argsort(pred, axis=1)[:, -1]
    pred_score = np.sort(pred, axis=1)[:, -1]

    precisions = []
    recalls = []
    f1_scores = []
    for thr in thrs:
        # Only prediction values larger than thr are counted as positive
        _pred_label = pred_label.copy()
        if thr is not None:
            _pred_label[pred_score <= thr] = -1
        pred_positive = label == _pred_label.reshape(-1, 1)
        gt_positive = label == target.reshape(-1, 1)
        precision = (pred_positive & gt_positive).sum(0) / np.maximum(
            pred_positive.sum(0), 1) * 100
        recall = (pred_positive & gt_positive).sum(0) / np.maximum(
            gt_positive.sum(0), 1) * 100
        f1_score = 2 * precision * recall / np.maximum(precision + recall,
                                                       1e-20)
        if average_mode == 'macro':
            precision = float(precision.mean())
            recall = float(recall.mean())
            f1_score = float(f1_score.mean())
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    if return_single:
        return precisions[0], recalls[0], f1_scores[0]
    else:
        return precisions, recalls, f1_scores