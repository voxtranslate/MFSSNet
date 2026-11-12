import hashlib
import logging
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

logger = logging.getLogger(__name__)


# ============================================================================
# DATASET
# ============================================================================

class BaseTextDataset(Dataset):
    """Pairs (corrupted, intact) of the TII-ST / TII-HT benchmarks.
    No mask is read: the network never receives one, and the damage target
    D* is computed from the pair itself."""
    def __init__(self, img_size=(64, 256), augment=False, config=None):
        self.img_size = img_size
        self.augment = augment
        self.config = config
        self.image_pairs = []
        # Threshold of the damage target D* (see __getitem__)
        self.damage_tau = getattr(config, 'damage_tau', 0.1)
        # Gaussian-noise augmentation of the training input. 0 = benchmark
        # protocol (same training data as the compared methods); > 0 only for
        # the model used in the separate robustness study.
        self.noise_prob = getattr(config, 'noise_prob', 0.0)
        # Noise added to TEST inputs, for the separate robustness study only.
        # 0 = benchmark protocol (test images used as distributed).
        self.test_noise_sigma = getattr(config, 'test_noise_sigma', 0.0)

        base_transforms = [transforms.Resize(img_size), transforms.ToTensor()] if img_size else [transforms.ToTensor()]
        self.transform = transforms.Compose(base_transforms)

    def apply_corruption(self, image):
        """Gaussian noise, sigma ~ U[0.05, 0.15] (training augmentation)"""
        noise = torch.randn_like(image) * random.uniform(0.05, 0.15)
        return torch.clamp(image + noise, 0, 1)

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        corrupted = self.transform(Image.open(pair['corrupted']).convert('RGB'))
        clean = self.transform(Image.open(pair['clean']).convert('RGB'))

        # Damage target D* (training-only supervision of the damage head):
        # pixels where the corrupted input differs from the intact image,
        # computed before any noise; no annotation needed.
        damage = ((corrupted - clean).abs().amax(dim=0, keepdim=True) > self.damage_tau).float()

        if self.augment and self.noise_prob > 0 and random.random() < self.noise_prob:
            corrupted = self.apply_corruption(corrupted)
        elif not self.augment and self.test_noise_sigma > 0:
            # Deterministic per image, so that every method sees the same noisy input
            g = torch.Generator().manual_seed(idx)
            corrupted = torch.clamp(corrupted + self.test_noise_sigma * torch.randn(corrupted.shape, generator=g), 0, 1)

        return {
            'corrupted': torch.clamp(corrupted, 0, 1),
            'clean': torch.clamp(clean, 0, 1),
            'damage': damage,
            'image_id': pair['id'],
            'name': pair['name'],
            'source': pair['source'],
            'dataset_type': pair['type']
        }


def _as_list(dirs):
    if dirs is None:
        return []
    return [dirs] if isinstance(dirs, (str, Path)) else list(dirs)


class MergedTextDataset(BaseTextDataset):
    """TII-ST (scene text: synthetic + real) and/or TII-HT (handwritten) pairs.
    `synthetic_dir` and `handwritten_dir` accept one folder or a list of folders,
    e.g. every source of the official TII-ST training set."""

    def __init__(self, synthetic_dir=None, handwritten_dir=None, split='train',
                 augment=False, config=None, img_size=None):
        super().__init__(img_size, augment, config)
        self.counts = {}
        for d in _as_list(synthetic_dir):
            self._load_synthetic(Path(d), split)
        for d in _as_list(handwritten_dir):
            self._load_handwritten(Path(d))

    def _add(self, img_path, clean_path, root, kind):
        self.image_pairs.append({
            'id': f"{kind}_{root.name}_{img_path.stem}",
            'name': img_path.name,
            'source': root.name,
            'corrupted': str(img_path),
            'clean': str(clean_path),
            'type': kind
        })
        self.counts[str(root)] = self.counts.get(str(root), 0) + 1

    def _load_synthetic(self, root_dir, split):
        if split == 'train':
            incomplete_dir, complete_dir = root_dir / 'train_incomplete', root_dir / 'train_f'
        else:
            incomplete_dir, complete_dir = root_dir / 't_incomplete', root_dir / 't_f'
        if not incomplete_dir.exists() and (root_dir / 'Corrupted_Image').exists():
            # the synthetic part of TII-ST (syth80k) is released with the TII-HT layout
            incomplete_dir, complete_dir = root_dir / 'Corrupted_Image', root_dir / 'Intact_Image'
        if not incomplete_dir.exists():
            logger.warning(f"Scene-text directory not found: {incomplete_dir}")
            return
        missing = 0
        for img_path in sorted(incomplete_dir.glob('*.png')):
            complete_path = complete_dir / img_path.name
            if complete_path.exists():
                self._add(img_path, complete_path, root_dir, 'scene')
            else:
                missing += 1
        if missing:
            logger.warning(f"{missing} corrupted images without intact counterpart in {root_dir}")

    def _load_handwritten(self, root_dir):
        corrupted_dir, intact_dir = root_dir / 'Corrupted_Image', root_dir / 'Intact_Image'
        if not corrupted_dir.exists():
            logger.warning(f"Handwritten directory not found: {corrupted_dir}")
            return
        missing = 0
        for img_path in sorted(corrupted_dir.glob('*.png')):
            intact_path = intact_dir / img_path.name
            if intact_path.exists():
                self._add(img_path, intact_path, root_dir, 'handwritten')
            else:
                missing += 1
        if missing:
            logger.warning(f"{missing} corrupted images without intact counterpart in {root_dir}")


# ============================================================================
# BENCHMARK PROTOCOL (Zhu et al., AAAI 2024): splits, seeds, sanity checks
# ============================================================================

# Official split sizes of the TII benchmarks
OFFICIAL_COUNTS = {
    'st_train': 80000 + 4877,   # synthetic + real (ICDAR 2013/2015/2017)
    'st_test': 1599,            # real
    'ht_train': 38578,          # IAM
    'ht_test': 1600,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_counts(dataset, key, strict=True):
    """Compare the number of loaded pairs with the official split size."""
    n, expected = len(dataset), OFFICIAL_COUNTS[key]
    msg = f"{key}: {n} pairs loaded (official: {expected}); per folder: {dataset.counts}"
    if n != expected:
        if strict:
            raise ValueError(msg + " -> the split differs from the benchmark; "
                                   "add the missing source folders or set config.strict_counts = False")
        logger.warning(msg)
    else:
        logger.info(msg)


def split_indices(n, val_fraction=0.02, seed=42):
    """Deterministic train/validation split of the TRAINING set (the official
    test set is never used for training or model selection)."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(val_fraction * n)))
    return sorted(idx[n_val:]), sorted(idx[:n_val])


def split_manifest(train_set, val_set, test_set):
    """Images of each split (source/name) with a SHA-256 fingerprint of each
    list, saved with every run so that the exact splits can be published."""
    def ids(ds):
        base, idx = (ds.dataset, ds.indices) if isinstance(ds, Subset) else (ds, range(len(ds)))
        return [f"{base.image_pairs[i]['source']}/{base.image_pairs[i]['name']}" for i in idx]
    out = {}
    for name, ds in (('train', train_set), ('validation', val_set), ('test', test_set)):
        lst = ids(ds)
        out[name] = {'n_images': len(lst), 'sha256': hashlib.sha256('\n'.join(lst).encode()).hexdigest(),
                     'images': lst}
    return out


def _md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def build_benchmark_datasets(config):
    """Training / validation / test sets of ONE benchmark (config.train_on =
    'st' or 'ht'), as in the protocol of the compared methods."""
    kind = config.train_on
    dirs = {'st': (config.st_train_dirs, config.st_test_dirs),
            'ht': (config.ht_train_dirs, config.ht_test_dirs)}[kind]
    kw = dict(img_size=config.img_size, config=config)
    as_arg = (lambda d: dict(synthetic_dir=d)) if kind == 'st' else (lambda d: dict(handwritten_dir=d))

    train_aug = MergedTextDataset(split='train', augment=True, **as_arg(dirs[0]), **kw)
    train_plain = MergedTextDataset(split='train', augment=False, **as_arg(dirs[0]), **kw)
    test_set = MergedTextDataset(split='test', augment=False, **as_arg(dirs[1]), **kw)
    strict = getattr(config, 'strict_counts', True)
    check_counts(train_aug, f'{kind}_train', strict)
    check_counts(test_set, f'{kind}_test', strict)

    tr_idx, va_idx = split_indices(len(train_aug), config.val_fraction, config.seed)
    train_set = Subset(train_aug, tr_idx)
    val_set = Subset(train_plain, va_idx)

    # Guard against leakage: no test image may also be a training image
    if getattr(config, 'check_leakage', True):
        train_hashes = {_md5(p['corrupted']) for p in train_aug.image_pairs}
        overlap = sum(_md5(p['corrupted']) in train_hashes for p in test_set.image_pairs)
        if overlap:
            logger.warning(f"{overlap} test images are byte-identical to training images")
        else:
            logger.info("No test image found in the training set")
    return train_set, val_set, test_set


# ============================================================================
# JOINT TII-ST + TII-HT TRAINING SET (real pairs only)
# ============================================================================
# Only the real, distributed pairs of the two benchmarks are used: nothing is
# generated, simulated or randomly altered (no noise augmentation, no mask
# synthesis; the damage target D* is computed from the pair as in
# BaseTextDataset). Each item has exactly the keys and values of an item of
# MergedTextDataset: corrupted, clean, damage, image_id, name, source,
# dataset_type ('scene' for TII-ST, 'handwritten' for TII-HT).
# The official test sets stay SEPARATE (different protocols): a joint model is
# evaluated on TII-ST test and on TII-HT test, one at a time.

JOINT_DOMAIN = {'scene': 'st', 'handwritten': 'ht'}


class JointTextDataset(MergedTextDataset):
    """TII-ST (synthetic + real folders) and TII-HT (IAM) pairs in one dataset."""

    def __init__(self, st_dirs=None, ht_dirs=None, split='train', config=None, img_size=None, verify=True):
        if config is not None and (getattr(config, 'noise_prob', 0.0) > 0 or getattr(config, 'test_noise_sigma', 0.0) > 0):
            raise ValueError("joint training uses the real pairs as distributed: set config.noise_prob = 0 and "
                             "config.test_noise_sigma = 0")
        super().__init__(synthetic_dir=st_dirs, handwritten_dir=ht_dirs, split=split, augment=False,
                         config=config, img_size=img_size)
        ids = [p['id'] for p in self.image_pairs]
        if len(set(ids)) != len(ids):                       # two folders with the same name: make the ids unique
            seen, dup = {}, 0
            for p in self.image_pairs:
                k = p['id']
                seen[k] = seen.get(k, 0) + 1
                if seen[k] > 1:
                    p['id'] = f"{k}#{hashlib.md5(p['corrupted'].encode()).hexdigest()[:8]}"
                    dup += 1
            logger.warning(f"{dup} image ids were not unique across folders and got a path hash")
        if verify:
            self.verify_pairs()

    def verify_pairs(self):
        """Reads the header of every file (no pixel decoding): both images of a pair must open, be RGB-convertible
        and have the same size. A pair that fails is a data error: the dataset stops instead of dropping it."""
        bad = []
        for p in self.image_pairs:
            try:
                with Image.open(p['corrupted']) as a, Image.open(p['clean']) as b:
                    if a.size != b.size:
                        bad.append((p['id'], f"size {a.size} vs {b.size}"))
            except Exception as e:                              # noqa: BLE001
                bad.append((p['id'], f"{type(e).__name__}: {e}"))
        if bad:
            raise ValueError(f"{len(bad)} corrupted/intact pairs are inconsistent, e.g. {bad[:3]}")

    def domain(self, i):
        """'st' or 'ht' of pair i."""
        return JOINT_DOMAIN[self.image_pairs[i]['type']]

    def domain_indices(self):
        out = {'st': [], 'ht': []}
        for i, p in enumerate(self.image_pairs):
            out[JOINT_DOMAIN[p['type']]].append(i)
        return out

    def summary(self):
        di = self.domain_indices()
        return {'st': len(di['st']), 'ht': len(di['ht']), 'total': len(self), 'per_folder': dict(self.counts)}


def joint_split_indices(dataset, val_fraction=0.02, seed=42):
    """Deterministic validation split drawn INSIDE each benchmark (so both are represented in the validation set).
    Only the training pairs are split: the official test sets are never used for training or model selection."""
    tr, va = [], []
    for dom, idx in dataset.domain_indices().items():
        t_, v_ = split_indices(len(idx), val_fraction, seed)
        tr += [idx[i] for i in t_]
        va += [idx[i] for i in v_]
    return sorted(tr), sorted(va)


def joint_sampler(train_set, mode='none', seed=42):
    """None (default): every real pair is seen exactly once per epoch.
    'equal': a seeded WeightedRandomSampler that gives each benchmark the same share of every epoch by re-drawing
    REAL pairs (the smaller benchmark is seen more than once per epoch; no image is created)."""
    if mode in (None, 'none'):
        return None
    if mode != 'equal':
        raise ValueError("mode must be 'none' or 'equal'")
    base, idx = (train_set.dataset, train_set.indices) if isinstance(train_set, Subset) else (train_set, range(len(train_set)))
    doms = [JOINT_DOMAIN[base.image_pairs[i]['type']] for i in idx]
    n = {d: doms.count(d) for d in ('st', 'ht')}
    weights = torch.tensor([1.0 / n[d] for d in doms], dtype=torch.double)
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(doms), replacement=True, generator=g)


def build_joint_datasets(config):
    """(train_set, val_set, test_sets) of the joint ST + HT training.
    train_set / val_set: Subsets of one JointTextDataset (both benchmarks); test_sets: {'st': ..., 'ht': ...}, the two
    official test sets, each evaluated with its own protocol."""
    kw = dict(config=config, img_size=config.img_size)
    full = JointTextDataset(config.st_train_dirs, config.ht_train_dirs, split='train', **kw)
    di = full.domain_indices()
    strict = getattr(config, 'strict_counts', True)
    for dom in ('st', 'ht'):
        n, expected = len(di[dom]), OFFICIAL_COUNTS[f'{dom}_train']
        msg = f"{dom}_train: {n} pairs loaded (official: {expected})"
        if n != expected:
            if strict:
                raise ValueError(msg + " -> the split differs from the benchmark; add the missing source folders "
                                       "or set config.strict_counts = False")
            logger.warning(msg)
        else:
            logger.info(msg)
    tests = {'st': MergedTextDataset(synthetic_dir=config.st_test_dirs, split='test', augment=False, **kw),
             'ht': MergedTextDataset(handwritten_dir=config.ht_test_dirs, split='test', augment=False, **kw)}
    for dom, ds in tests.items():
        check_counts(ds, f'{dom}_test', strict)
    tr_idx, va_idx = joint_split_indices(full, config.val_fraction, config.seed)
    train_set, val_set = Subset(full, tr_idx), Subset(full, va_idx)
    if getattr(config, 'check_leakage', True):
        train_hashes = {_md5(p['corrupted']) for p in full.image_pairs}
        for dom, ds in tests.items():
            overlap = sum(_md5(p['corrupted']) in train_hashes for p in ds.image_pairs)
            if overlap:
                logger.warning(f"{dom}_test: {overlap} test images are byte-identical to joint training images")
            else:
                logger.info(f"{dom}_test: no test image found in the joint training set")
    logger.info(f"joint set: {full.summary()}; train {len(train_set)}, validation {len(val_set)}")
    return train_set, val_set, tests
