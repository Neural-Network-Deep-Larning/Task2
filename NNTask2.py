import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(layout="wide", page_title="MLP Backprop GUI (Penguins)")


# -------------------------
# Utility / Data loading / Pre Processing  will be imported from our last task
# -------------------------
@st.cache_data



# -------------------------
# Placeholder for your team's backprop implementation
# -------------------------

class CustomNN:
    def __init__(self):
        self.initialized = False

    def initialize(self, num_of_layers, num_neurons, use_bias=True, activation='sigmoid', lr=0.01, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)

        self.num_features = 5
        self.num_classes = 3
        self.use_bias = use_bias
        self.activation_name = activation
        self.lr = lr

        # Handle user input for layers
        if isinstance(num_neurons, int):
            num_neurons = [num_neurons] * num_of_layers
        if len(num_neurons) != num_of_layers:
            raise ValueError("Length of num_neurons list must match num_of_layers.")

        # Choose activation
        if activation == 'sigmoid':
            self.activation = lambda x: 1 / (1 + np.exp(-x))
            self.activation_derivative = lambda a: a * (1 - a)
        elif activation == 'tanh':
            self.activation = np.tanh
            self.activation_derivative = lambda a: 1 - np.square(a)
        else:
            raise ValueError("Unsupported activation. Choose 'sigmoid' or 'tanh'.")

        # Initialize weights and biases
        self.weights, self.biases = [], []
        layer_dims = [self.num_features] + num_neurons + [self.num_classes]

        for i in range(len(layer_dims) - 1):
            w = np.random.randn(layer_dims[i], layer_dims[i + 1]) * 0.01
            self.weights.append(w)
            if use_bias:
                b = np.zeros((1, layer_dims[i + 1]))
                self.biases.append(b)
            else:
                self.biases.append(np.zeros((1, layer_dims[i + 1])))

        self.initialized = True
        st.success("✅ Network initialized successfully!")

    # ---------- Forward Pass ----------
    def _forward(self, X):
        activations = [X]

        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            a = self.activation(z)
            activations.append(a)

        # Output layer with softmax
        z_out = activations[-1] @ self.weights[-1] + self.biases[-1]
        exp_scores = np.exp(z_out - np.max(z_out, axis=1, keepdims=True))
        a_out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        activations.append(a_out)

        return activations

    # ---------- Backpropagation ----------
    def _backward(self, activations, y_true):
        grads_w, grads_b = [None] * len(self.weights), [None] * len(self.biases)

        m = y_true.shape[0]
        y_one_hot = np.zeros((m, self.num_classes))
        y_one_hot[np.arange(m), y_true] = 1

        # Output layer gradient
        delta = activations[-1] - y_one_hot
        grads_w[-1] = activations[-2].T @ delta / m
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True) / m

        # Hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            delta = (delta @ self.weights[i + 1].T) * self.activation_derivative(activations[i + 1])
            grads_w[i] = activations[i].T @ delta / m
            grads_b[i] = np.sum(delta, axis=0, keepdims=True) / m

        return grads_w, grads_b

    # ---------- Training ----------
    def train(self, X_train, y_train, epochs=100, on_epoch=None):
        if not self.initialized:
            st.error("Network not initialized. Call initialize() first.")
            return

        losses, accs = [], []

        for epoch in range(epochs):
            # Forward
            activations, zs = self._forward(X_train)
            y_pred = activations[-1]

            # Loss (Cross-Entropy)
            m = y_train.shape[0]
            log_likelihood = -np.log(y_pred[range(m), y_train] + 1e-9)
            loss = np.sum(log_likelihood) / m
            losses.append(loss)

            # Accuracy
            preds = np.argmax(y_pred, axis=1)
            acc = np.mean(preds == y_train)
            accs.append(acc)

            # Backpropagation
            grads_w, grads_b = self._backward(activations, zs, y_train)

            # Update weights
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * grads_w[i]
                self.biases[i] -= self.lr * grads_b[i]

            if on_epoch:
                on_epoch(epoch, loss, acc)

        st.success("✅ Training complete.")
        return {'loss': losses, 'train_acc': accs}

    # ---------- Prediction ----------
    def predict(self, X):
        if not self.initialized:
            st.error("❌ Network not initialized. Call initialize() first.")
            return

        activations, _ = self._forward(X)
        y_pred = np.argmax(activations[-1], axis=1)
        return y_pred


# -------------------------
# Streamlit UI
# -------------------------
st.title('Backprop MLP GUI — Penguins classification')
st.markdown('This GUI lets you configure and test your custom MLP implementation on the Penguins dataset.\n\nYour team should complete the backprop logic inside the **CustomNN** class above.')

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header('Network configuration')
    num_hidden_layers = st.number_input('Number of hidden layers', min_value=0, max_value=5, value=1, step=1)
    hidden_neurons_raw = st.text_input('Neurons per hidden layer (comma separated)', value='5')
    learning_rate = st.number_input('Learning rate', min_value=1e-6, value=0.01, format="%g")
    epochs = st.number_input('Number of epochs', min_value=1, value=500, step=1)
    add_bias = st.checkbox('Add bias', value=True)
    activation_choice = st.selectbox('Activation function', options=['sigmoid', 'tanh'])
    random_state = st.number_input('Random seed', value=42)

# Parse neuron counts safely
try:
    hidden_sizes = [int(x.strip()) for x in hidden_neurons_raw.split(',') if x.strip() != '']
except Exception:
    st.error('Could not parse neurons per layer. Use comma separated integers like "4,5"')
    hidden_sizes = []

# Load data
X_all, y_all, df_all = load_penguins_dataset()
feature_cols = X_all.columns.tolist()

st.subheader('Dataset')
st.write('Features used:', feature_cols)
st.write('Number of samples loaded:', len(y_all))

# Split per species: 30 train + 20 test per class
species_list = y_all.unique().tolist()
train_idx, test_idx = [], []
for sp in species_list:
    idxs = df_all[df_all['Species'] == sp].index.tolist()
    train_idx.extend(idxs[:30])
    test_idx.extend(idxs[30:50])

X_train = X_all.loc[train_idx].reset_index(drop=True)
X_test = X_all.loc[test_idx].reset_index(drop=True)
y_train = y_all.loc[train_idx].reset_index(drop=True)
y_test = y_all.loc[test_idx].reset_index(drop=True)

st.write('Training samples per class: 30 — total', len(y_train))
st.write('Test samples per class: 20 — total', len(y_test))

# Scale data
# TODO: Scaling will be imported from our last task

# Encode labels
label_to_int = {lab: i for i, lab in enumerate(species_list)}
int_to_label = {i: lab for lab, i in label_to_int.items()}
y_train_int = y_train.map(label_to_int)
y_test_int = y_test.map(label_to_int)

model = CustomNN()

# ============================================================
# Main Controls (TODOs for your team)
# ============================================================
col1, col2 = st.columns([2, 3])
with col1:
    st.header('Controls')

    # -----------------------------
    # INITIALIZE MODEL
    # -----------------------------
    if st.button('Initialize model'):
        layer_sizes = [X_train.shape[1]] + hidden_sizes + [len(species_list)]
        
        # TODO: Inside CustomNN.initialize()
        # - Initialize weights and biases according to layer_sizes.
        # - Apply random seed if provided.
        # - Support chosen activation and bias inclusion.
        # - Set learning rate and mark model as initialized.
        model.initialize(
            layer_sizes=layer_sizes,
            use_bias=add_bias,
            activation=activation_choice,
            lr=learning_rate,
            random_state=int(random_state)
        )
        st.success('Model initialized (placeholder).')

    # -----------------------------
    # TRAIN MODEL
    # -----------------------------
    if st.button('Train model'):
        if not model.initialized:
            st.error('Model not initialized. Click "Initialize model" first.')
        else:
            progress = st.progress(0)
            status = st.empty()

            def on_epoch(e, logs):
                if e % max(1, int(epochs / 100)) == 0:
                    progress.progress(int(100 * e / epochs))
                    status.text(f'Epoch {e+1}/{epochs} — loss: {logs.get("loss")} — train_acc: {logs.get("train_acc")}')

            # TODO: Inside CustomNN.train()
            # - Implement full forward + backward propagation training loop.
            # - Compute loss and update weights using gradients.
            # - Report loss and accuracy periodically via on_epoch().
            model.train(X_train_scaled.values, y_train_int.values, epochs=int(epochs), on_epoch=on_epoch)

            st.success('Training completed (placeholder).')

    # -----------------------------
    # TEST MODEL
    # -----------------------------
    if st.button('Test model'):
        if not model.initialized:
            st.error('Model not initialized.')
        else:
            # TODO: Inside CustomNN.predict()
            # - Run forward pass to get predicted classes (argmax of outputs).
            y_pred_int = model.predict(X_test_scaled.values)

            acc = accuracy_score(y_test_int, y_pred_int)
            st.metric('Test accuracy', f'{acc * 100:.2f}%')

            # TODO (Optional): Add model evaluation saving or visualization. will be imported from our last task
            cm = confusion_matrix(y_test_int, y_pred_int, labels=list(range(len(species_list))))
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=species_list).plot(ax=ax, colorbar=False)
            st.pyplot(fig)

# ============================================================
# Right Column — Visualizations
# ============================================================
with col2:
    st.header('Visualizations')
    if st.checkbox('Show class distribution'):
        fig, ax = plt.subplots()
        y_all.value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel('Species')
        ax.set_ylabel('Count')
        st.pyplot(fig)

# ============================================================
# Single Sample Classification
# ============================================================
st.header('Classify a single sample')
manual_values = {}
cols = st.columns(5)
for i, col in enumerate(cols):
    val = col.number_input(f'{feature_cols[i]}', value=float(X_test.iloc[0, i]))
    manual_values[feature_cols[i]] = val

sample = pd.DataFrame([manual_values])[feature_cols]
if st.button('Classify sample'):
    sample_scaled = pd.DataFrame(scaler.transform(sample), columns=feature_cols)
    if not model.initialized:
        st.error('Model not initialized.')
    else:
        pred_int = model.predict(sample_scaled.values)[0]
        pred_label = int_to_label.get(int(pred_int), 'Unknown')
        st.success(f'Predicted class: {pred_label}')

# ============================================================
# Footer
# ============================================================
st.markdown('---')
st.info('This version contains only the GUI and placeholders.\nYour team should fill in the `CustomNN` class to complete the backprop implementation.')
