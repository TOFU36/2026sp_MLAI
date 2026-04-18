import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from src.utils.tools import set_seed, get_device
from src.utils.logger import save_json
from src.data.loader import load_train_test, create_dataloaders
from src.models import ResNet1D
from src.training.trainer import ECGTrainer
from src.evaluation.embeddings import get_embeddings


def main():
    set_seed(42)
    device = get_device()
    train_df, test_df = load_train_test()
    train_loader, test_loader = create_dataloaders(train_df, test_df)

    print("=" * 50 + "\nExperiment 3.3: Classifier Decoupling (Softmax vs ML)\n" + "=" * 50)

    # Pre-train feature extractor
    print(">>> Pre-training 1D-ResNet feature extractor ...")
    model = ResNet1D(use_se=False)
    trainer = ECGTrainer(model, train_loader, test_loader, device,
                         'results/models/Phase3_3_Extractor')
    trainer.fit(epochs=10)

    base_f1 = trainer.evaluate()['f1']

    # Extract embeddings
    print(">>> Extracting embeddings ...")
    X_train_emb, y_train_emb = get_embeddings(model, train_loader, device)
    X_test_emb, y_test_emb = get_embeddings(model, test_loader, device)

    # Compare classifiers
    classifiers = {
        "End-to-End Softmax": None,
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "SVM (RBF Kernel)": SVC(kernel='rbf', class_weight='balanced', random_state=42),
    }

    results = {"End-to-End Softmax": base_f1}
    print(f"  -> End-to-End Softmax F1: {base_f1:.4f}")

    for name, clf in classifiers.items():
        if clf is None:
            continue
        print(f">>> {name} on embeddings ...")
        clf.fit(X_train_emb, y_train_emb)
        score = f1_score(y_test_emb, clf.predict(X_test_emb), average='macro')
        results[name] = score
        print(f"  -> {name} F1: {score:.4f}")

    save_json(results, 'results/logs/phase3_3_classifiers.json')
    print("\nPhase 3.3 done.")


if __name__ == "__main__":
    main()
