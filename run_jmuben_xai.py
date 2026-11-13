#!/usr/bin/env python3
"""Generate XAI outputs from completed JMuBEN checkpoints without retraining."""
import argparse
import json
from pathlib import Path
from torchvision import datasets
from train_jmuben_hybrid import run_xai


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--results', required=True)
    ap.add_argument('--image-size', type=int, default=160)
    ap.add_argument('--samples-per-class', type=int, default=5)
    args = ap.parse_args()
    results = Path(args.results)
    base = datasets.ImageFolder(args.data_root)
    split = json.loads((results / 'split_manifest.json').read_text())
    test_indices = split['test']
    selection = json.loads((results / 'validation_selected_model.json').read_text())
    selected = selection['selected_model']
    models = [selected]
    if 'HybridEfficientMobile' not in models:
        models.append('HybridEfficientMobile')
    for name in models:
        checkpoint = results / f'{name}_best.pt'
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        print(f'Generating Grad-CAM, Integrated Gradients, and Occlusion for {name}', flush=True)
        run_xai(name, checkpoint, base, test_indices, base.classes,
                args.image_size, results, args.samples_per_class)
    (results / 'xai_models.json').write_text(json.dumps({
        'validation_selected_model': selected,
        'additional_predeclared_model': 'HybridEfficientMobile',
        'models_explained': models,
        'note': 'Model selection uses validation macro-F1; the hybrid is explained as the predeclared fusion architecture, not chosen using test scores.'
    }, indent=2))
    print('XAI recovery complete:', results.resolve(), flush=True)


if __name__ == '__main__':
    main()
