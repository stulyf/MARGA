#!/usr/bin/env python
import os
import re
import argparse
import pickle
import numpy as np
import pandas as pd


def load_dist_matrix(file_path: str) -> np.ndarray:
    # TELIKA: first sheet, skip header row, first column as index (labels), second column are addresses to drop
    df = pd.read_excel(file_path, sheet_name=0, skiprows=1, index_col=0)
    df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    return df.values


def trilaterate_two_anchors(distance_matrix: np.ndarray) -> np.ndarray:
    """Reconstruct 2D coords from distance matrix using two anchors (0 at (0,0), 1 at (d01,0))."""
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    coords = np.zeros((n, 2), dtype=float)
    coords[0] = (0.0, 0.0)
    d01 = float(D[0, 1]) if n > 1 else 0.0
    coords[1] = (d01, 0.0)
    if d01 == 0:
        # fallback: spread points on small circle to avoid division by zero
        for i in range(2, n):
            angle = 2 * np.pi * (i - 1) / max(1, n - 1)
            coords[i] = (np.cos(angle) * 1e-3, np.sin(angle) * 1e-3)
        return coords
    for i in range(2, n):
        d_i0 = float(D[0, i])
        d_i1 = float(D[1, i])
        x = (d_i0 ** 2 - d_i1 ** 2 + d01 ** 2) / (2.0 * d01)
        y2 = d_i0 ** 2 - x ** 2
        y = np.sqrt(y2) if y2 > 0 else 0.0
        coords[i] = (x, y)
    return coords


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    cmin = coords.min(axis=0)
    cmax = coords.max(axis=0)
    span = np.where((cmax - cmin) < 1e-12, 1.0, (cmax - cmin))
    return (coords - cmin) / span


def save_dataset(dataset, filename):
    d = os.path.dirname(filename)
    os.makedirs(d, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description='Convert TELIKA DATA to TOP pkl using two-anchor trilateration + augmentation')
    parser.add_argument('--src_dir', required=True, help='Directory with TELIKA excel files')
    parser.add_argument('--out_root', default='data_real', help='Output root directory')
    parser.add_argument('--output_name', default='telika_aug', help='Subfolder name under out_root')
    parser.add_argument('--num_augments', type=int, default=999, help='Number of noisy augmentations per instance')
    parser.add_argument('--noise_level', type=float, default=0.1, help='Gaussian noise std added to coords')
    parser.add_argument('--max_length', type=float, default=2.0, help='Normalized max length L')
    args = parser.parse_args()

    src = os.path.abspath(args.src_dir)
    out_root = os.path.abspath(args.out_root)
    output_path = os.path.join(out_root, args.output_name)
    files = [f for f in os.listdir(src) if f.lower().endswith('.xlsx') and not f.startswith('~$')]
    # Skip parameter doc
    files = [f for f in files if 'values of parameters' not in f.lower()]

    # Group by number of nodes parsed from filename suffix _N.xlsx
    grouped = {}
    for f in sorted(files):
        m = re.search(r'_(\d+)\.xlsx$', f, re.IGNORECASE)
        if not m:
            continue
        n = int(m.group(1))
        grouped.setdefault(n, []).append(os.path.join(src, f))

    for num_nodes, paths in grouped.items():
        depots, nodes, prizes, lengths = [], [], [], []
        for p in paths:
            D = load_dist_matrix(p)
            coords = trilaterate_two_anchors(D)
            coords = normalize_coords(coords)
            depot = coords[0]
            loc = coords[1:]
            depots.append(depot)
            nodes.append(loc)
            prizes.append(np.ones(num_nodes, dtype=float))
            lengths.append(float(args.max_length))

            # augmentations
            for _ in range(args.num_augments):
                noise = np.random.normal(0.0, args.noise_level, size=(num_nodes + 1, 2))
                augmented = coords + noise
                augmented = normalize_coords(augmented)
                depots.append(augmented[0])
                nodes.append(augmented[1:])
                prizes.append(np.ones(num_nodes, dtype=float))
                lengths.append(float(args.max_length))

        # pack
        output = [np.array(depots), np.array(nodes), np.array(prizes), np.array(lengths)]
        dataset = list(zip(*[item.tolist() for item in output]))
        out_file = os.path.join(output_path, f'TELIKA_{num_nodes}.pkl')
        save_dataset(dataset, out_file)
        print(f'[OK] Saved {len(dataset)} instances with {num_nodes} nodes -> {out_file}')


if __name__ == '__main__':
    main()


