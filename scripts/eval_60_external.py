import os
import random
import requests
from collections import defaultdict

def run_evaluation():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'external_validation', 'organised')
    classes = {
        'glioma': 'GLIOMA',
        'meningioma': 'MENINGIOMA',
        'notumor': 'HEALTHY',
        'pituitary': 'PITUITARY'
    }

    all_samples = []
    # Seed for reproducible random sampling
    random.seed(42)

    for folder, target in classes.items():
        folder_path = os.path.join(data_dir, folder)
        if os.path.exists(folder_path):
            files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            selected = random.sample(files, min(15, len(files)))
            for s in selected:
                all_samples.append((s, target, folder))

    random.shuffle(all_samples)
    total = len(all_samples)
    print(f"\n================================================================================")
    print(f"   NEUROSCAN AI: EXTERNAL VALIDATION BATCH TEST ({total} RANDOM CASES)")
    print(f"================================================================================\n")

    correct = 0
    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)
    confidences = []
    failures = []

    print(f"| {'#':<3} | {'Image File':<28} | {'Ground Truth':<12} | {'Prediction':<12} | {'Conf (%)':<8} | {'Status':<7} |")
    print("|" + "-"*5 + "|" + "-"*30 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*10 + "|" + "-"*9 + "|")

    for idx, (filepath, ground_truth, folder) in enumerate(all_samples, 1):
        fname = os.path.basename(filepath)
        try:
            with open(filepath, 'rb') as f:
                res = requests.post('http://127.0.0.1:8000/predict', files={'file': f}, timeout=10)
            if res.status_code == 200:
                data = res.json()
                pred = data.get('diagnosis')
                conf = data.get('confidence', 0.0)
                confidences.append(conf)
            else:
                pred = f"ERR_{res.status_code}"
                conf = 0.0
        except Exception as e:
            pred = "CONN_ERR"
            conf = 0.0

        is_correct = (pred == ground_truth)
        if is_correct:
            correct += 1
            per_class_correct[ground_truth] += 1
        else:
            failures.append((fname, ground_truth, pred, conf))

        per_class_total[ground_truth] += 1
        status_str = "PASS [OK]" if is_correct else "FAIL [X]"
        print(f"| {idx:<3} | {fname:<28} | {ground_truth:<12} | {pred:<12} | {conf:<8.2f} | {status_str:<7} |")

    acc = (correct / total) * 100 if total > 0 else 0.0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    print("\n" + "="*80)
    print("                      ACCURACY & PERFORMANCE SUMMARY")
    print("="*80)
    print(f"  • Total External Cases Evaluated : {total}")
    print(f"  • Correctly Classified Slices    : {correct} / {total}")
    print(f"  • Overall Validation Accuracy    : {acc:.2f}%")
    print(f"  • Mean Diagnostic Confidence     : {avg_conf:.2f}%\n")

    print("  Per-Class Breakdown:")
    for gt in ['GLIOMA', 'MENINGIOMA', 'PITUITARY', 'HEALTHY']:
        tot = per_class_total[gt]
        corr = per_class_correct[gt]
        pct = (corr / tot * 100) if tot > 0 else 0.0
        print(f"    - {gt:<12}: {corr:2d} / {tot:2d} ({pct:5.1f}%)")

    if failures:
        print(f"\n  Misclassified Cases ({len(failures)}):")
        for f_name, gt, pr, cf in failures:
            print(f"    - File: {f_name:<25} | Expected: {gt:<10} | Predicted: {pr:<10} | Conf: {cf:.2f}%")
    else:
        print("\n  [PERFECT] 100% Accuracy: Zero misclassifications across all 60 external test cases!")

if __name__ == '__main__':
    run_evaluation()
