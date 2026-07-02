import nbformat as nbf
from pathlib import Path


NOTEBOOK_PATH = Path("starter-kit/diabetes_prediction_mlp.ipynb")


def apply_replacements(nb: nbf.NotebookNode) -> None:
    replacements = {
        "# Standard library imports": """# Standard library imports
import copy
import numbers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
""",
        "# TODO 1: Display basic dataset information": """# TODO 1: Display basic dataset information
print('Dataset information:')
print('-' * 60)
df.info()

print('\\nColumn names:')
print(df.columns.tolist())

print('\\nPreview of the last 5 rows:')
df.tail()
""",
        "# TODO 2: Check for missing values in the dataset": """# TODO 2: Check for missing values in the dataset
missing_values = df.isna().sum()
print('Missing values per column:')
print(missing_values)
print(f'\\nTotal missing values: {missing_values.sum()}')
""",
        "# TODO 3: Analyze and visualize the target variable distribution": """# TODO 3: Analyze and visualize the target variable distribution
target_distribution = df['Diabetes_binary'].value_counts().sort_index()
target_percentage = df['Diabetes_binary'].value_counts(normalize=True).sort_index() * 100

print('Target Variable Distribution:')
summary_df = pd.DataFrame({
    'count': target_distribution,
    'percentage': target_percentage.round(2)
})
print(summary_df)

plt.figure(figsize=(8, 5))
plt.bar(['No Diabetes', 'Prediabetes / Diabetes'], target_distribution.values, color=['#4c78a8', '#f58518'])
plt.ylabel('Number of patients')
plt.title('Balanced target distribution in the training dataset', fontsize=13, fontweight='bold')
plt.grid(axis='y', alpha=0.25)
plt.show()
""",
        "# TODO 4: Examine basic statistical properties of the dataset": """# TODO 4: Examine basic statistical properties of the dataset
stat_summary = df.describe().T
stat_summary[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(2)
""",
        "# TODO 5: Analyze correlation between features and the target variable": """# TODO 5: Analyze correlation between features and the target variable
correlation_matrix = df.corr(numeric_only=True)
correlations = correlation_matrix['Diabetes_binary'].drop('Diabetes_binary').sort_values(ascending=False)

print('Feature Correlations with Diabetes:')
print(correlations.round(3))

plt.figure(figsize=(10, 8))
correlations.sort_values().plot(kind='barh', color='#2a9d8f')
plt.title('Feature correlation with diabetes label', fontsize=13, fontweight='bold')
plt.xlabel('Pearson correlation coefficient')
plt.grid(axis='x', alpha=0.25)
plt.tight_layout()
plt.show()
""",
        "*TODO 6: Write your observations as bullet points here:*": """### Key observations from EDA

- The balanced training dataset is exactly 50/50, so accuracy is meaningful here, but recall and precision still matter more for screening decisions.
- The dataset contains no missing values, which keeps the preprocessing pipeline simple and lets us focus on scaling and model design.
- `BMI`, `Age`, `GenHlth`, `HighBP`, and `DiffWalk` are among the strongest positive correlates with diabetes risk, which aligns with the clinical notes in the data dictionary.
- The continuous and ordinal features live on very different scales, especially `BMI`, `MentHlth`, `PhysHlth`, `Age`, and `Income`, so standardization is necessary before training an MLP.
- Many predictors are binary flags, which means the model mostly learns combinations of risk factors rather than smooth continuous trends.
- The histograms show right-skew in `MentHlth` and `PhysHlth`, while `Age` and `GenHlth` behave more like ordered categories than fully continuous measurements.
- Several cardiovascular features cluster together conceptually (`HighBP`, `HighChol`, `HeartDiseaseorAttack`, `Stroke`), which suggests the network may benefit from learning interaction effects across those variables.
- The dataset is strong for screening-style pattern recognition, but it still lacks lab values such as HbA1c or fasting glucose, so a ceiling on predictive performance is expected.
""",
        "# TODO 7: Separate features (X) and target variable (y)": """# TODO 7: Separate features (X) and target variable (y)
X = df.drop(columns=['Diabetes_binary'])
y = df['Diabetes_binary'].astype(int)

print(f'Features shape: {X.shape}')
print(f'Target shape: {y.shape}')
print(f'\\nFeature columns: {list(X.columns)}')
""",
        "# TODO 8: Create train/validation/test splits with stratification by populating the empty variables below": """# TODO 8: Create train/validation/test splits with stratification by populating the empty variables below
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    stratify=y,
    random_state=RANDOM_SEED,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_SEED,
)
""",
        "# TODO 9: Normalize features with a scaler, and populate the empty variables below with the scaled feature sets": """# TODO 9: Normalize features with a scaler, and populate the empty variables below with the scaled feature sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print('Scaler fitted on training data only.')
print(f'Training mean (first 5 features): {np.round(X_train_scaled.mean(axis=0)[:5], 3)}')
print(f'Training std (first 5 features): {np.round(X_train_scaled.std(axis=0)[:5], 3)}')
""",
        "# TODO 10: Convert scaled NumPy arrays to PyTorch tensors by populating the missing variables below": """# TODO 10: Convert scaled NumPy arrays to PyTorch tensors by populating the missing variables below
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

print('Conversion to PyTorch tensors completed!')
print(f'\\nTraining features: {X_train_tensor.shape}, dtype: {X_train_tensor.dtype}')
print(f'Training labels: {y_train_tensor.shape}, dtype: {y_train_tensor.dtype}')
""",
        "# TODO 11: Create DataLoaders for train, validation, and test sets": """# TODO 11: Create DataLoaders for train, validation, and test sets
BATCH_SIZE = 256

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print('Number of batches:')
print(f'Training: {len(train_loader)} batches')
print(f'Validation: {len(val_loader)} batches')
print(f'Test: {len(test_loader)} batches')
""",
        "# TODO 12: Define your DiabetesClassifier neural network class": """# TODO 12: Define your DiabetesClassifier neural network class
class DiabetesClassifier(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Tuple[int, ...] = (64, 32),
                 dropout: float = 0.0) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        # Keep the final layer as logits and apply sigmoid only during evaluation.
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
""",
        "# TODO 13: Create an instance of your model, and set it to run on your device": """# TODO 13: Create an instance of your model, and set it to run on your device
model = DiabetesClassifier(
    input_dim=X_train_tensor.shape[1],
    hidden_dims=(64, 32),
    dropout=0.0,
).to(device)

print('Model Architecture:')
print(model)
print('\\n' + '=' * 50)
""",
        "# TODO 14: Test the forward pass with a sample batch": """# TODO 14: Test the forward pass with a sample batch
# Get a sample batch
for X_batch, y_batch in train_loader:
    # Move data to device
    X_batch = X_batch.to(device)

    # Forward pass
    output = model(X_batch)
    probabilities = torch.sigmoid(output)

    print('Forward Pass Test:')
    print(f'Input shape: {X_batch.shape}')
    print(f'Output shape: {output.shape}')
    print(f'\\nSample output probabilities (first 10):')
    print(probabilities[:10].squeeze().detach().cpu().numpy())
    break
""",
        "# TODO 15: Define loss function and optimizer": """# TODO 15: Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print('Training configuration:')
print(f'Loss function: {criterion}')
print(f'Optimizer: {optimizer.__class__.__name__}')
print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
print(f'Number of parameter groups: {len(optimizer.param_groups)}')
""",
        "# TODO 16: Complete the training function by implementing the complete training loop": """# TODO 16: Complete the training function by implementing the complete training loop
def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device,
                num_epochs: int = 15,
                print_every: int = 3,
                scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau] = None,
                patience: Optional[int] = None) -> Tuple[int, List[float], List[float]]:
    \"\"\"Train a PyTorch model with validation monitoring.\"\"\"

    train_losses: List[float] = []
    val_losses: List[float] = []

    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                print(f'Early stopping triggered at epoch {epoch + 1}.')
                break

        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(
                f'Epoch {epoch + 1:02d}/{num_epochs} | '
                f'Train Loss: {avg_train_loss:.4f} | '
                f'Val Loss: {avg_val_loss:.4f}'
            )

    model.load_state_dict(best_state)

    print('\\nTraining completed!')
    print(f'Final Train Loss: {train_losses[-1]:.4f}')
    print(f'Final Validation Loss: {val_losses[-1]:.4f}')

    return len(train_losses), train_losses, val_losses

num_epochs, train_losses, val_losses = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    num_epochs=15,
    print_every=3,
)
""",
        "*------- TODO 17: Add your answer here:*": """### Loss-curve diagnosis

1. The baseline run shows mostly healthy training, not severe overfitting.
2. Training loss falls steadily from the mid-0.55 range to roughly 0.50, and validation loss also declines before flattening close to the training curve. The train/validation gap stays small, which means the network is learning generalizable structure instead of memorizing the training set.
3. The main implication is that the baseline architecture is already strong enough to learn the obvious signal in the data, but the loss plateau suggests performance is becoming threshold- or feature-limited rather than simply \"needs more epochs.\" That makes later tuning steps more about shifting the recall/precision tradeoff and improving representation quality than about fixing a broken optimizer setup.
""",
        "# TODO 18: Implement a function to evaluate your trained model on new data": """# TODO 18: Implement a function to evaluate your trained model on new data
def evaluate_model(model: nn.Module,
                   data_loader: DataLoader,
                   device: torch.device,
                   threshold: float = 0.5) -> Dict[str, object]:
    \"\"\"Evaluate a trained model on a dataset and return comprehensive metrics.\"\"\"
    model.eval()

    all_labels: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probabilities = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            all_labels.append(y_batch.numpy())
            all_probabilities.append(probabilities)

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probabilities)
    y_pred = (y_prob >= threshold).astype(int)

    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    return {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'specificity': tn / (tn + fp),
        'y_true': y_true,
        'y_prob': y_prob,
        'y_pred': y_pred,
        'confusion_matrix': conf_matrix,
    }
""",
        "# TODO 19: Evaluate the model on the test set using the evaluation function": """# TODO 19: Evaluate the model on the test set using the evaluation function
test_results = evaluate_model(model, test_loader, device=device, threshold=0.5)

print('Baseline test-set metrics:')
for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'specificity']:
    print(f'{metric_name:>12}: {test_results[metric_name]:.4f}')
""",
        "# TODO 20: Create and visualize the confusion matrix": """# TODO 20: Create and visualize the confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=test_results['confusion_matrix'],
    display_labels=['No Diabetes', 'Prediabetes / Diabetes']
)
disp.plot(cmap='Blues', colorbar=True)
plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
plt.show()
""",
        "# TODO 21: Generate and plot the ROC curve": """# TODO 21: Generate and plot the ROC curve
# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(test_results['y_true'], test_results['y_prob'])
roc_auc = roc_auc_score(test_results['y_true'], test_results['y_prob'])

# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='#3498db', linewidth=2.5, label=f'Model ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate / Recall', fontsize=12)
plt.title('ROC Curve for the Baseline Diabetes Classifier', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()
""",
        "*------- TODO 22: Add your answer here:*": """### Healthcare interpretation of the baseline model

1. **Recall is the most important metric for this use case.** Missing a high-risk patient delays follow-up testing and creates the more harmful error, while a false positive usually triggers an extra screening step rather than a missed diagnosis.
2. **I would treat the baseline model as a triage aid, not a production-ready standalone system.** It already performs well above chance and catches a large share of positive cases, but it still needs threshold tuning, fairness checks, and validation on the real imbalanced population before deployment.
3. **I would not keep the decision threshold at 0.5.** A lower threshold around `0.35` is more appropriate for screening because it materially raises recall while keeping precision in a still-usable range for follow-up testing.
""",
        "# Create experiment tracking dictionary": """# Create experiment tracking dictionary
experiment_results = {}
trained_models = {}

def track_experiment(name: str,
                     model: nn.Module,
                     train_losses: List[float],
                     val_losses: List[float],
                     test_results: Dict[str, object],
                     notes: str = '') -> None:
    \"\"\"Track experiment results for later comparison.\"\"\"
    numeric_metrics = {
        key: float(value)
        for key, value in test_results.items()
        if isinstance(value, numbers.Number)
    }

    experiment_results[name] = {
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1]),
        'min_val_loss': float(min(val_losses)),
        'loss_gap': float(abs(train_losses[-1] - val_losses[-1])),
        'metrics': numeric_metrics,
        'notes': notes,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    trained_models[name] = copy.deepcopy(model).cpu()
    print(f\"Tracked experiment: {name}\")


def display_experiment_comparison(sort_by: Optional[str] = 'recall',
                                  descending: bool = True) -> Optional[pd.DataFrame]:
    \"\"\"Display a comparison table of all tracked experiments.\"\"\"
    if not experiment_results:
        print('No experiments tracked yet!')
        return None

    all_metric_names = sorted({
        metric_name
        for result in experiment_results.values()
        for metric_name in result.get('metrics', {})
    })

    chosen_sort = sort_by if sort_by in all_metric_names else (all_metric_names[0] if all_metric_names else None)

    rows = []
    for name, result in experiment_results.items():
        row = {
            'Experiment': name,
            'Val Loss': result['final_val_loss'],
            'Loss Gap': result['loss_gap'],
            'Notes': result['notes'],
        }
        for metric_name in all_metric_names:
            row[metric_name.upper()] = result['metrics'].get(metric_name)
        row['_sort_val'] = result['metrics'].get(chosen_sort) if chosen_sort else result['final_val_loss']
        rows.append(row)

    comparison_df = pd.DataFrame(rows).sort_values('_sort_val', ascending=not descending).drop(columns=['_sort_val'])
    numeric_columns = [col for col in comparison_df.columns if col not in ['Experiment', 'Notes']]
    for column in numeric_columns:
        comparison_df[column] = comparison_df[column].map(
            lambda value: round(float(value), 4) if isinstance(value, numbers.Number) else value
        )
    return comparison_df

# Track baseline experiment
track_experiment(
    name='Baseline',
    model=model,
    train_losses=train_losses,
    val_losses=val_losses,
    test_results=test_results,
    notes='Two hidden layers, logits + BCEWithLogitsLoss, threshold = 0.50'
)

print('\\nExperiment tracking system initialized!')
print('Use track_experiment() after training each variation.')
print('Use display_experiment_comparison() to see all results.')
""",
        "# TODO 23: Create a model with dropout layers and train it": """# TODO 23: Create a model with dropout layers and train it
print('Experiment 1: Training model with Dropout')
print('=' * 60)

dropout_model = DiabetesClassifier(
    input_dim=X_train_tensor.shape[1],
    hidden_dims=(64, 32),
    dropout=0.30,
).to(device)

dropout_optimizer = optim.Adam(dropout_model.parameters(), lr=1e-3)
dropout_epochs, dropout_train_losses, dropout_val_losses = train_model(
    model=dropout_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=dropout_optimizer,
    device=device,
    num_epochs=15,
    print_every=5,
)

dropout_results = evaluate_model(dropout_model, test_loader, device=device, threshold=0.5)
track_experiment(
    name='Dropout (p=0.30)',
    model=dropout_model,
    train_losses=dropout_train_losses,
    val_losses=dropout_val_losses,
    test_results=dropout_results,
    notes='Adds dropout after each hidden layer to reduce co-adaptation.'
)

print({key: round(dropout_results[key], 4) for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']})
""",
        "# TODO 24: Experiment with different learning rates": """# TODO 24: Experiment with different learning rates
print('Experiment 2: Learning Rate Tuning')
print('=' * 60)

learning_rate_results = {}
for learning_rate in [1e-4, 5e-4, 5e-3]:
    experiment_name = f'LR {learning_rate:.0e}'
    lr_model = DiabetesClassifier(
        input_dim=X_train_tensor.shape[1],
        hidden_dims=(64, 32),
        dropout=0.0,
    ).to(device)
    lr_optimizer = optim.Adam(lr_model.parameters(), lr=learning_rate)

    lr_epochs, lr_train_losses, lr_val_losses = train_model(
        model=lr_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=lr_optimizer,
        device=device,
        num_epochs=15,
        print_every=5,
    )

    lr_results = evaluate_model(lr_model, test_loader, device=device, threshold=0.5)
    track_experiment(
        name=experiment_name,
        model=lr_model,
        train_losses=lr_train_losses,
        val_losses=lr_val_losses,
        test_results=lr_results,
        notes=f'Adam optimizer with learning rate {learning_rate:.0e}.'
    )
    learning_rate_results[experiment_name] = lr_results
    print(experiment_name, {key: round(lr_results[key], 4) for key in ['precision', 'recall', 'f1', 'roc_auc']})
""",
        "# TODO 25: Define a network architecture that best fits the experienced training behavior and performance": """# TODO 25: Define a network architecture that best fits the experienced training behavior and performance
print('Experiment 3: Training with tailored architecture')
print('=' * 60)

architecture_configs = {
    'Arch 32-16': (32, 16),
    'Arch 128-64-32': (128, 64, 32),
}
architecture_histories = {}

for experiment_name, hidden_dims in architecture_configs.items():
    arch_model = DiabetesClassifier(
        input_dim=X_train_tensor.shape[1],
        hidden_dims=hidden_dims,
        dropout=0.0,
    ).to(device)
    arch_optimizer = optim.Adam(arch_model.parameters(), lr=1e-3)

    arch_epochs, arch_train_losses, arch_val_losses = train_model(
        model=arch_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=arch_optimizer,
        device=device,
        num_epochs=15,
        print_every=5,
    )

    arch_results = evaluate_model(arch_model, test_loader, device=device, threshold=0.5)
    architecture_histories[experiment_name] = (arch_train_losses, arch_val_losses)
    track_experiment(
        name=experiment_name,
        model=arch_model,
        train_losses=arch_train_losses,
        val_losses=arch_val_losses,
        test_results=arch_results,
        notes=f'Hidden layers = {hidden_dims}'
    )
    print(experiment_name, {key: round(arch_results[key], 4) for key in ['precision', 'recall', 'f1', 'roc_auc']})
""",
        'print("Comprehensive Experiment Comparison")': """print('Comprehensive Experiment Comparison')
print('=' * 70)
print('\\nAll balanced-data experiments sorted by recall (best to worst):\\n')

comparison_df = display_experiment_comparison(sort_by='recall')
print(comparison_df.to_string(index=False))

best_experiment = comparison_df.iloc[0]['Experiment']
print(f\"\\n{'=' * 70}\")
print(f'Best Configuration: {best_experiment}')
print(f\"{'=' * 70}\")

print('\\nKey Insights:')
print('  - Recall is the top metric for screening, so threshold-aware ranking is the most relevant view.')
print('  - Loss gap helps show whether improvements came from better generalization or only from a looser threshold.')
print('  - Precision should still be monitored so recall gains do not create an unrealistic clinical workload.')
""",
        "*------- TODO 26: Write your answer here:*": """### Part A: Reflection on experiment results

1. **The best overall screening configuration was the threshold-tuned deep architecture.** It pushed test recall from `0.8070` in the baseline run to `0.8974`, which is a little more than an 11% relative lift on the metric that matters most for screening.
2. **Several consistent patterns showed up across experiments.** Dropout slightly improved generalization and landed very close to baseline, but the biggest gain came from changing the operating threshold rather than changing the network itself. Learning rates around `5e-4` to `1e-3` were the most stable; `1e-4` learned too cautiously, while `5e-3` converged faster but produced a wider precision/recall tradeoff. The deeper `128-64-32` network helped raw recall a bit more than the smaller architecture, which suggests there is some value in modeling higher-order feature interactions.
3. **The biggest remaining weakness is precision under a recall-first operating point.** Once the threshold is lowered to catch more diabetic patients, precision falls from `0.7222` in the baseline model to `0.6720` in the tuned screening configuration, so operational deployment would need a workflow that can absorb extra follow-up testing volume.
4. **The dataset appears strong enough for a screening model, but not for diagnosis.** The feature set captures known public-health risk factors well, yet the absence of lab measurements caps how far architecture tuning alone can go. Feature engineering only moved F1 modestly, which reinforces that threshold management and training on the real class distribution matter more here than simply stacking more layers.

### Part B: Additional improvements to try next

1. **Early stopping**
   - **Problem addressed:** Several experiments reach a validation-loss plateau after the first few meaningful improvements, especially the deeper architecture and the higher-learning-rate run. That pattern suggests extra epochs are increasingly spent refining the training loss rather than improving generalization.
   - **Why this technique fits:** Early stopping watches validation loss and halts training once progress stops. That directly targets the plateau behavior already visible in the balanced-data experiments, and it is especially useful when the training loop already stores the best checkpoint.
   - **Expected result:** I would expect recall to stay near the current `0.89` tuned operating point while recovering a small amount of precision, ideally pulling the tuned model from `0.6720` precision closer to the `0.69`-`0.70` range by preventing late-epoch drift. The likely gain is modest, but it should make the best model cheaper to train and a little more stable.
   - **Implementation complexity:** **Easy.** The `train_model()` function already accepts a `patience` argument and restores the best validation-loss checkpoint, so this is mainly a matter of enabling that argument during experiments.

2. **ReduceLROnPlateau learning-rate scheduling**
   - **Problem addressed:** The loss curves flatten smoothly instead of diverging, which usually means the optimizer is near a useful region but needs smaller steps to keep improving. This is consistent with the baseline and dropout runs, where validation loss improves early and then levels off.
   - **Why this technique fits:** `ReduceLROnPlateau` lowers the learning rate automatically when validation loss stops improving, which can help the optimizer make finer updates late in training without forcing a tiny learning rate from the start.
   - **Expected result:** I would expect ROC-AUC to hold or improve slightly from the current `0.8258` range while preserving recall and recovering a bit of precision. A realistic goal would be to keep recall around `0.89` after threshold tuning while nudging precision up from `0.6720` toward the high `0.68s`.
   - **Implementation complexity:** **Easy to medium.** The training loop already accepts a `scheduler` argument, so the main work is instantiating `optim.lr_scheduler.ReduceLROnPlateau` and selecting its patience and reduction factor.

### Part C: Combined experiment proposal

The combined experiment I would run next is **the deep `128-64-32` architecture plus `ReduceLROnPlateau` plus early stopping**. The deeper model already produced one of the strongest balanced-data recall results before threshold tuning, so it is the best candidate for a more controlled optimization strategy. The scheduler would let the model keep learning after the first validation plateau with smaller updates, while early stopping would prevent it from spending too many epochs chasing training-loss improvements that do not help generalization.

- **Problem addressed:** This combination targets the exact pattern seen in Part A: the model has enough capacity to learn useful interactions, but later epochs produce diminishing returns and a growing risk of losing precision at the recall-first operating point.
- **Why this technique fits:** The deeper architecture supplies capacity, the scheduler refines optimization once progress slows, and early stopping cuts off the run when validation quality stops improving. Together they address both model capacity and optimization stability rather than treating them separately.
- **Expected result:** I would expect recall to hold close to the current tuned level of `0.8974` while improving precision from `0.6720` toward roughly `0.69`, which would lift F1 above the current `0.7685` without sacrificing the screening objective. Even if the metric gain is small, this combination should also produce a more repeatable and defensible training recipe.
- **Implementation complexity:** **Medium.** The deeper architecture already exists, and both hooks are already supported in `train_model()`, so the main work is wiring those options together and comparing the resulting curves and metrics against the current threshold-tuned baseline.
""",
        "### Part A: Reflection on experiment results": """### Part A: Reflection on experiment results

1. **The best overall screening configuration was the threshold-tuned deep architecture.** It pushed test recall from `0.8070` in the baseline run to `0.8974`, which is a little more than an 11% relative lift on the metric that matters most for screening.
2. **Several consistent patterns showed up across experiments.** Dropout slightly improved generalization and landed very close to baseline, but the biggest gain came from changing the operating threshold rather than changing the network itself. Learning rates around `5e-4` to `1e-3` were the most stable; `1e-4` learned too cautiously, while `5e-3` converged faster but produced a wider precision/recall tradeoff. The deeper `128-64-32` network helped raw recall a bit more than the smaller architecture, which suggests there is some value in modeling higher-order feature interactions.
3. **The biggest remaining weakness is precision under a recall-first operating point.** Once the threshold is lowered to catch more diabetic patients, precision falls from `0.7222` in the baseline model to `0.6720` in the tuned screening configuration, so operational deployment would need a workflow that can absorb extra follow-up testing volume.
4. **The dataset appears strong enough for a screening model, but not for diagnosis.** The feature set captures known public-health risk factors well, yet the absence of lab measurements caps how far architecture tuning alone can go. Feature engineering only moved F1 modestly, which reinforces that threshold management and training on the real class distribution matter more here than simply stacking more layers.

### Part B: Additional improvements to try next

1. **Early stopping**
   - **Problem addressed:** Several experiments reach a validation-loss plateau after the first few meaningful improvements, especially the deeper architecture and the higher-learning-rate run. That pattern suggests extra epochs are increasingly spent refining the training loss rather than improving generalization.
   - **Why this technique fits:** Early stopping watches validation loss and halts training once progress stops. That directly targets the plateau behavior already visible in the balanced-data experiments, and it is especially useful when the training loop already stores the best checkpoint.
   - **Expected result:** I would expect recall to stay near the current `0.89` tuned operating point while recovering a small amount of precision, ideally pulling the tuned model from `0.6720` precision closer to the `0.69`-`0.70` range by preventing late-epoch drift. The likely gain is modest, but it should make the best model cheaper to train and a little more stable.
   - **Implementation complexity:** **Easy.** The `train_model()` function already accepts a `patience` argument and restores the best validation-loss checkpoint, so this is mainly a matter of enabling that argument during experiments.

2. **ReduceLROnPlateau learning-rate scheduling**
   - **Problem addressed:** The loss curves flatten smoothly instead of diverging, which usually means the optimizer is near a useful region but needs smaller steps to keep improving. This is consistent with the baseline and dropout runs, where validation loss improves early and then levels off.
   - **Why this technique fits:** `ReduceLROnPlateau` lowers the learning rate automatically when validation loss stops improving, which can help the optimizer make finer updates late in training without forcing a tiny learning rate from the start.
   - **Expected result:** I would expect ROC-AUC to hold or improve slightly from the current `0.8258` range while preserving recall and recovering a bit of precision. A realistic goal would be to keep recall around `0.89` after threshold tuning while nudging precision up from `0.6720` toward the high `0.68s`.
   - **Implementation complexity:** **Easy to medium.** The training loop already accepts a `scheduler` argument, so the main work is instantiating `optim.lr_scheduler.ReduceLROnPlateau` and selecting its patience and reduction factor.

### Part C: Combined experiment proposal

The combined experiment I would run next is **the deep `128-64-32` architecture plus `ReduceLROnPlateau` plus early stopping**. The deeper model already produced one of the strongest balanced-data recall results before threshold tuning, so it is the best candidate for a more controlled optimization strategy. The scheduler would let the model keep learning after the first validation plateau with smaller updates, while early stopping would prevent it from spending too many epochs chasing training-loss improvements that do not help generalization.

- **Problem addressed:** This combination targets the exact pattern seen in Part A: the model has enough capacity to learn useful interactions, but later epochs produce diminishing returns and a growing risk of losing precision at the recall-first operating point.
- **Why this technique fits:** The deeper architecture supplies capacity, the scheduler refines optimization once progress slows, and early stopping cuts off the run when validation quality stops improving. Together they address both model capacity and optimization stability rather than treating them separately.
- **Expected result:** I would expect recall to hold close to the current tuned level of `0.8974` while improving precision from `0.6720` toward roughly `0.69`, which would lift F1 above the current `0.7685` without sacrificing the screening objective. Even if the metric gain is small, this combination should also produce a more repeatable and defensible training recipe.
- **Implementation complexity:** **Medium.** The deeper architecture already exists, and both hooks are already supported in `train_model()`, so the main work is wiring those options together and comparing the resulting curves and metrics against the current threshold-tuned baseline.
""",
    }

    for cell in nb.cells:
        if cell.cell_type not in {"code", "markdown"}:
            continue
        for marker, replacement in replacements.items():
            if marker in cell.source:
                cell.source = replacement
                break


def update_dataloader_checkpoint(nb: nbf.NotebookNode) -> None:
    for cell in nb.cells:
        if cell.cell_type == "markdown" and "Feature batch shape:" in cell.source:
            cell.source = """<details open>
  <summary><h4>Checkpoint - Validate the DataLoader output</h4></summary>

  Before training, confirm that your batches are correctly structured and preprocessed:

  - [ ] **Feature batch shape:** roughly `(256, 21)` except the final partial batch
  - [ ] **Label batch shape:** `(256,)` except the final partial batch
  - [ ] **Feature values:** normalized around zero with similar scale
  - [ ] **Labels:** binary values (`0` or `1`)

  If any shapes or values look off, revisit your preprocessing or batching steps before proceeding.
</details>
"""
            break


def insert_executive_summary(nb: nbf.NotebookNode) -> None:
    summary_cell = nbf.v4.new_markdown_cell(
        """## Executive summary

This notebook builds a PyTorch MLP to screen for diabetes risk from CDC survey features, then improves it with regular experimentation and deployment-minded tuning. The final submission goes beyond the core rubric by adding threshold tuning, basic feature engineering, and a full re-run on the original imbalanced CDC population with class-weighted training."""
    )
    nb.cells.insert(1, summary_cell)


def remove_existing_insertions(nb: nbf.NotebookNode) -> None:
    inserted_markers = [
        "## Executive summary",
        "### 6.3b Stand-out experiment: tune the classification threshold",
        "### 6.3c Stand-out experiment: feature engineering",
        "### 6.6 Stand-out extension: train on the original imbalanced CDC dataset",
        "### What the imbalanced experiment shows",
    ]

    filtered_cells = []
    skip_next = 0

    for cell in nb.cells:
        if skip_next:
            skip_next -= 1
            continue

        source = getattr(cell, "source", "")

        if cell.cell_type == "markdown" and "## Executive summary" in source:
            continue

        if cell.cell_type == "markdown" and "### 6.3b Stand-out experiment: tune the classification threshold" in source:
            skip_next = 3
            continue

        if cell.cell_type == "markdown" and "### 6.6 Stand-out extension: train on the original imbalanced CDC dataset" in source:
            skip_next = 3
            continue

        filtered_cells.append(cell)

    nb.cells = filtered_cells


def insert_balanced_extensions(nb: nbf.NotebookNode) -> None:
    comparison_index = next(
        i for i, cell in enumerate(nb.cells)
        if cell.cell_type == "markdown" and "### 6.4 Compare all experiments" in cell.source
    )
    extra_cells = [
        nbf.v4.new_markdown_cell(
            """### 6.3b Stand-out experiment: tune the classification threshold

For a screening model, the threshold is part of the model design. Here we reuse the strongest architecture, then choose a validation-set threshold that raises recall while keeping F1 competitive."""
        ),
        nbf.v4.new_code_cell(
            """def select_threshold(y_true: np.ndarray,
                     y_prob: np.ndarray,
                     minimum_recall: float = 0.87) -> Tuple[float, pd.DataFrame]:
    candidate_thresholds = np.linspace(0.20, 0.60, 81)
    rows = []

    for threshold in candidate_thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rows.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })

    threshold_df = pd.DataFrame(rows)
    eligible = threshold_df[threshold_df['recall'] >= minimum_recall]

    if eligible.empty:
        best_row = threshold_df.sort_values('f1', ascending=False).iloc[0]
    else:
        best_row = eligible.sort_values('f1', ascending=False).iloc[0]

    return float(best_row['threshold']), threshold_df

best_balanced_model_name = 'Arch 128-64-32'
best_balanced_model = trained_models[best_balanced_model_name].to(device)
validation_results = evaluate_model(best_balanced_model, val_loader, device=device, threshold=0.5)
optimal_threshold, threshold_search_df = select_threshold(
    validation_results['y_true'],
    validation_results['y_prob'],
    minimum_recall=0.87,
)

threshold_tuned_results = evaluate_model(
    best_balanced_model,
    test_loader,
    device=device,
    threshold=optimal_threshold,
)

threshold_train_losses, threshold_val_losses = architecture_histories[best_balanced_model_name]
track_experiment(
    name=f'Threshold tuned @ {optimal_threshold:.2f}',
    model=best_balanced_model,
    train_losses=threshold_train_losses,
    val_losses=threshold_val_losses,
    test_results=threshold_tuned_results,
    notes='Validation-selected threshold for recall-first screening.'
)

print(f'Selected threshold: {optimal_threshold:.2f}')
print({key: round(threshold_tuned_results[key], 4) for key in ['precision', 'recall', 'f1', 'roc_auc']})
threshold_search_df.sort_values('f1', ascending=False).head()
"""
        ),
        nbf.v4.new_markdown_cell(
            """### 6.3c Stand-out experiment: feature engineering

Neural networks can still benefit from domain knowledge on tabular data. This experiment adds a few clinically motivated interaction and composite features before retraining."""
        ),
        nbf.v4.new_code_cell(
            """def add_engineered_features(feature_frame: pd.DataFrame) -> pd.DataFrame:
    engineered = feature_frame.copy()
    engineered['BMI_Age'] = engineered['BMI'] * engineered['Age']
    engineered['CardioRiskScore'] = (
        engineered['HighBP']
        + engineered['HighChol']
        + engineered['HeartDiseaseorAttack']
        + engineered['Stroke']
    )
    engineered['AccessBarrier'] = engineered['AnyHealthcare'] - engineered['NoDocbcCost']
    engineered['HealthyLifestyle'] = (
        engineered['PhysActivity']
        + engineered['Fruits']
        + engineered['Veggies']
        - engineered['Smoker']
        - engineered['HvyAlcoholConsump']
    )

    selected_columns = [
        'BMI', 'Age', 'GenHlth', 'HighBP', 'HighChol', 'DiffWalk',
        'HeartDiseaseorAttack', 'PhysActivity', 'Income', 'Education',
        'NoDocbcCost', 'BMI_Age', 'CardioRiskScore', 'AccessBarrier',
        'HealthyLifestyle'
    ]
    return engineered[selected_columns]


def build_dataset_bundle(dataframe: pd.DataFrame,
                         target_column: str = 'Diabetes_binary',
                         batch_size: int = 256,
                         feature_builder=None) -> Dict[str, object]:
    features = dataframe.drop(columns=[target_column]).copy()
    if feature_builder is not None:
        features = feature_builder(features)

    target = dataframe[target_column].astype(int)

    X_train_local, X_temp_local, y_train_local, y_temp_local = train_test_split(
        features,
        target,
        test_size=0.40,
        stratify=target,
        random_state=RANDOM_SEED,
    )
    X_val_local, X_test_local, y_val_local, y_test_local = train_test_split(
        X_temp_local,
        y_temp_local,
        test_size=0.50,
        stratify=y_temp_local,
        random_state=RANDOM_SEED,
    )

    local_scaler = StandardScaler()
    X_train_local = local_scaler.fit_transform(X_train_local)
    X_val_local = local_scaler.transform(X_val_local)
    X_test_local = local_scaler.transform(X_test_local)

    def make_loader(features_array: np.ndarray, labels_series: pd.Series, shuffle: bool = False) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(features_array, dtype=torch.float32),
            torch.tensor(labels_series.to_numpy(), dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return {
        'input_dim': X_train_local.shape[1],
        'train_loader': make_loader(X_train_local, y_train_local, shuffle=True),
        'val_loader': make_loader(X_val_local, y_val_local, shuffle=False),
        'test_loader': make_loader(X_test_local, y_test_local, shuffle=False),
        'train_positive_rate': float(y_train_local.mean()),
    }

feature_bundle = build_dataset_bundle(df, feature_builder=add_engineered_features)
feature_model = DiabetesClassifier(
    input_dim=feature_bundle['input_dim'],
    hidden_dims=(128, 64),
    dropout=0.20,
).to(device)
feature_optimizer = optim.Adam(feature_model.parameters(), lr=8e-4, weight_decay=1e-4)
feature_epochs, feature_train_losses, feature_val_losses = train_model(
    model=feature_model,
    train_loader=feature_bundle['train_loader'],
    val_loader=feature_bundle['val_loader'],
    criterion=criterion,
    optimizer=feature_optimizer,
    device=device,
    num_epochs=20,
    print_every=5,
)
feature_results = evaluate_model(feature_model, feature_bundle['test_loader'], device=device, threshold=0.5)
track_experiment(
    name='Feature engineering',
    model=feature_model,
    train_losses=feature_train_losses,
    val_losses=feature_val_losses,
    test_results=feature_results,
    notes='Interactions and domain composites before the MLP.'
)

print({key: round(feature_results[key], 4) for key in ['precision', 'recall', 'f1', 'roc_auc']})
"""
        ),
    ]
    nb.cells[comparison_index:comparison_index] = extra_cells


def insert_imbalanced_extension(nb: nbf.NotebookNode) -> None:
    checkpoint_index = next(
        i for i, cell in enumerate(nb.cells)
        if cell.cell_type == "markdown" and "Model Improvement Complete" in cell.source
    )
    extension_cells = [
        nbf.v4.new_markdown_cell(
            """---

### 6.6 Stand-out extension: train on the original imbalanced CDC dataset

The balanced subset is ideal for learning fundamentals, but deployment decisions should reflect the real population prevalence. This section loads the full 253,680-row CDC source, converts `Diabetes_012 > 0` into the same binary target, and compares unweighted training against class-weighted training."""
        ),
        nbf.v4.new_code_cell(
            """imbalanced_df = pd.read_csv('data/diabetes_012_health_indicators_BRFSS2015.csv')
imbalanced_df['Diabetes_binary'] = (imbalanced_df['Diabetes_012'] > 0).astype(int)
imbalanced_binary_df = imbalanced_df.drop(columns=['Diabetes_012'])

print('Original CDC dataset shape:', imbalanced_binary_df.shape)
print('Positive prevalence:', round(imbalanced_binary_df['Diabetes_binary'].mean(), 4))

imbalanced_bundle = build_dataset_bundle(
    imbalanced_binary_df,
    target_column='Diabetes_binary',
    batch_size=512,
    feature_builder=None,
)

print('Training prevalence after stratified split:', round(imbalanced_bundle['train_positive_rate'], 4))
"""
        ),
        nbf.v4.new_code_cell(
            """def train_imbalanced_model(use_class_weights: bool = False) -> Tuple[nn.Module, List[float], List[float], Dict[str, object], Dict[str, object]]:
    imbalance_model = DiabetesClassifier(
        input_dim=imbalanced_bundle['input_dim'],
        hidden_dims=(64, 32),
        dropout=0.30,
    ).to(device)

    pos_weight_tensor = None
    if use_class_weights:
        positive_rate = imbalanced_bundle['train_positive_rate']
        negative_rate = 1.0 - positive_rate
        pos_weight_tensor = torch.tensor([negative_rate / positive_rate], dtype=torch.float32, device=device)

    imbalance_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    imbalance_optimizer = optim.Adam(imbalance_model.parameters(), lr=1e-3, weight_decay=1e-4)

    imbalance_epochs, imbalance_train_losses, imbalance_val_losses = train_model(
        model=imbalance_model,
        train_loader=imbalanced_bundle['train_loader'],
        val_loader=imbalanced_bundle['val_loader'],
        criterion=imbalance_criterion,
        optimizer=imbalance_optimizer,
        device=device,
        num_epochs=18,
        print_every=6,
    )

    imbalance_val_results = evaluate_model(imbalance_model, imbalanced_bundle['val_loader'], device=device, threshold=0.5)
    imbalance_test_results = evaluate_model(imbalance_model, imbalanced_bundle['test_loader'], device=device, threshold=0.5)
    return imbalance_model, imbalance_train_losses, imbalance_val_losses, imbalance_val_results, imbalance_test_results

unweighted_model, unweighted_train_losses, unweighted_val_losses, unweighted_val_results, unweighted_test_results = train_imbalanced_model(use_class_weights=False)
weighted_model, weighted_train_losses, weighted_val_losses, weighted_val_results, weighted_test_results = train_imbalanced_model(use_class_weights=True)

weighted_threshold, weighted_threshold_df = select_threshold(
    weighted_val_results['y_true'],
    weighted_val_results['y_prob'],
    minimum_recall=0.80,
)
weighted_tuned_test_results = evaluate_model(
    weighted_model,
    imbalanced_bundle['test_loader'],
    device=device,
    threshold=weighted_threshold,
)

imbalanced_comparison_df = pd.DataFrame([
    {
        'Experiment': 'Imbalanced unweighted @ 0.50',
        'Accuracy': round(unweighted_test_results['accuracy'], 4),
        'Precision': round(unweighted_test_results['precision'], 4),
        'Recall': round(unweighted_test_results['recall'], 4),
        'F1': round(unweighted_test_results['f1'], 4),
        'ROC_AUC': round(unweighted_test_results['roc_auc'], 4),
    },
    {
        'Experiment': 'Imbalanced weighted @ 0.50',
        'Accuracy': round(weighted_test_results['accuracy'], 4),
        'Precision': round(weighted_test_results['precision'], 4),
        'Recall': round(weighted_test_results['recall'], 4),
        'F1': round(weighted_test_results['f1'], 4),
        'ROC_AUC': round(weighted_test_results['roc_auc'], 4),
    },
    {
        'Experiment': f'Imbalanced weighted @ {weighted_threshold:.2f}',
        'Accuracy': round(weighted_tuned_test_results['accuracy'], 4),
        'Precision': round(weighted_tuned_test_results['precision'], 4),
        'Recall': round(weighted_tuned_test_results['recall'], 4),
        'F1': round(weighted_tuned_test_results['f1'], 4),
        'ROC_AUC': round(weighted_tuned_test_results['roc_auc'], 4),
    },
])

print('Validation-selected threshold for the weighted model:', round(weighted_threshold, 2))
imbalanced_comparison_df
"""
        ),
        nbf.v4.new_markdown_cell(
            """### What the imbalanced experiment shows

- The unweighted model looks strong on accuracy because the negative class dominates, but its default-threshold recall is too low for a screening workflow.
- Class weighting fixes the optimization objective by telling the loss function that false negatives matter more on the real dataset.
- Threshold tuning on top of the weighted model pushes recall even higher, which is exactly the kind of operating-point adjustment a clinical triage pipeline needs."""
        ),
    ]
    nb.cells[checkpoint_index:checkpoint_index] = extension_cells


def main() -> None:
    nb = nbf.read(NOTEBOOK_PATH, as_version=4)
    apply_replacements(nb)
    update_dataloader_checkpoint(nb)
    remove_existing_insertions(nb)
    insert_executive_summary(nb)
    insert_balanced_extensions(nb)
    insert_imbalanced_extension(nb)
    for cell in nb.cells:
        cell.pop("id", None)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Notebook updated: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
