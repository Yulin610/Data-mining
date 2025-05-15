import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
import json
import os
from datetime import datetime
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import re  # 用于文本预处理
warnings.filterwarnings('ignore')

# Use simple tokenization method without NLTK dependency
def simple_tokenize(text):
    return text.lower().split()

class TextPreprocessor:
    def __init__(self):
        # Use basic stop words list
        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'of'])
        
    def preprocess(self, text):
        # Tokenization
        tokens = simple_tokenize(text)
        # Remove stop words
        tokens = [t for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)

class TraditionalModel:
    def __init__(self, model_type='lr'):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2
        )
        if model_type == 'lr':
            self.classifier = LogisticRegression(max_iter=1000, C=1.0, warm_start=True)
        elif model_type == 'nb':
            self.classifier = MultinomialNB()
        elif model_type == 'svm':
            self.classifier = LinearSVC(C=1.0, max_iter=2000)
        self.preprocessor = TextPreprocessor()
        self.model_dir = "models"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Add training history
        self.training_history = []
        # Add flag to track if vectorizer is fitted
        self.is_vectorizer_fitted = False
        
    def save_model(self):
        """Save model, vectorizer and training history using fixed filenames"""
        model_path = os.path.join(self.model_dir, f"{self.model_type}_model.pkl")
        vectorizer_path = os.path.join(self.model_dir, f"{self.model_type}_vectorizer.pkl")
        history_path = os.path.join(self.model_dir, f"{self.model_type}_history.json")
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f)
        
        print(f"Model saved to: {model_path}")
        print(f"Vectorizer saved to: {vectorizer_path}")
        print(f"Training history saved to: {history_path}")
        
    def load_model(self, model_path=None, vectorizer_path=None):
        """Load model, vectorizer and training history"""
        if model_path is None or vectorizer_path is None:
            # Use fixed filenames
            model_path = os.path.join(self.model_dir, f"{self.model_type}_model.pkl")
            vectorizer_path = os.path.join(self.model_dir, f"{self.model_type}_vectorizer.pkl")
            history_path = os.path.join(self.model_dir, f"{self.model_type}_history.json")
            
            # Check if files exist
            if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
                print("No saved model found, will use new model")
                return False
            
            # Load training history
            if os.path.exists(history_path):
                try:
                    with open(history_path, 'r') as f:
                        self.training_history = json.load(f)
                    print(f"Training history loaded: {history_path}")
                except Exception as e:
                    print(f"Error loading training history: {str(e)}")
        
        try:
            with open(model_path, 'rb') as f:
                self.classifier = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print(f"Model loaded: {model_path}")
            print(f"Vectorizer loaded: {vectorizer_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
        
    def train(self, X_train, y_train, incremental=True):
        """Train model with incremental training support"""
        # Preprocess text
        X_train_processed = [self.preprocessor.preprocess(text) for text in X_train]
        
        # Always fit vectorizer on new data to ensure proper vocabulary
        X_train_tfidf = self.vectorizer.fit_transform(X_train_processed)
        self.is_vectorizer_fitted = True
        
        # Train classifier
        self.classifier.fit(X_train_tfidf, y_train)
        
        # Record training information
        training_info = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'samples': len(X_train),
            'positive_ratio': sum(y_train == 1) / len(y_train),
            'model_params': self.classifier.get_params(),
            'vectorizer_fitted': True
        }
        self.training_history.append(training_info)
        
        # Save model
        self.save_model()
        
    def predict(self, X):
        if not self.is_vectorizer_fitted:
            raise ValueError("Vectorizer must be fitted before prediction")
            
        # Preprocess text
        X_processed = [self.preprocessor.preprocess(text) for text in X]
        # Convert to TF-IDF features
        X_tfidf = self.vectorizer.transform(X_processed)
        # Predict
        return self.classifier.predict(X_tfidf)
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        # Record evaluation results
        eval_info = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'accuracy': accuracy,
            'report': report
        }
        self.training_history[-1].update(eval_info)
        
        return accuracy, report, predictions

def save_results(model_name, accuracy, report, predictions=None):
    """Save model results to file using fixed filename"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results = {
        "model_name": model_name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "accuracy": accuracy,
        "classification_report": report
    }
    
    if predictions is not None:
        # Handle both numpy arrays and lists
        if hasattr(predictions, 'tolist'):
            results["predictions"] = predictions.tolist()
        else:
            # If predictions is already a list, use it directly
            results["predictions"] = predictions
    
    filename = f"{results_dir}/{model_name}_results.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to: {filename}")

def load_and_preprocess_data(file_path, sample_size=10000, start_index=0):
    """Load data, ensuring each sample contains both positive and negative reviews"""
    # Read data
    print(f"Reading data file: {file_path}")
    df = pd.read_csv(file_path)
    
    # Display column names
    print("Data columns:", df.columns.tolist())
    
    # Rename columns
    df = df.rename(columns={'text': 'review', 'label': 'sentiment'})
    
    # Display unique label values
    print("Unique label values:", df['sentiment'].unique())
    
    # Handle missing values
    print("Data shape before processing:", df.shape)
    df = df.dropna()  # Remove rows with NaN
    # Fill empty strings with a placeholder
    df['review'] = df['review'].fillna('')
    df['review'] = df['review'].astype(str)  # Convert all reviews to string
    print("Data shape after processing:", df.shape)
    
    # Convert sentiment labels to numeric values
    if df['sentiment'].dtype == 'object':
        # Display label distribution before conversion
        print("Label distribution before conversion:")
        print(df['sentiment'].value_counts())
        
        # Try different label mappings
        label_map = {
            'positive': 1, 'negative': 0,
            '积极': 1, '消极': 0,
            '1': 1, '0': 0,
            1: 1, 0: 0,
            'pos': 1, 'neg': 0,
            'POS': 1, 'NEG': 0
        }
        df['sentiment'] = df['sentiment'].map(label_map)
        
        # Display label distribution after conversion
        print("Label distribution after conversion:")
        print(df['sentiment'].value_counts())
    
    # Check for remaining NaN values
    if df['sentiment'].isna().any():
        print("Warning: NaN values still exist in sentiment labels, will be filled with 0")
        df['sentiment'] = df['sentiment'].fillna(0)
    
    # Get positive and negative samples separately
    positive_samples = df[df['sentiment'] == 1]
    negative_samples = df[df['sentiment'] == 0]
    
    # Calculate samples per class
    samples_per_class = sample_size // 2
    
    # Sample from each class
    if len(positive_samples) > samples_per_class:
        positive_samples = positive_samples.sample(n=samples_per_class, random_state=42)
    if len(negative_samples) > samples_per_class:
        negative_samples = negative_samples.sample(n=samples_per_class, random_state=42)
    
    # Combine positive and negative samples
    df_sampled = pd.concat([positive_samples, negative_samples])
    
    # Shuffle data
    df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Data distribution after sampling:")
    print(df_sampled['sentiment'].value_counts())
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df_sampled['review'].values,
        df_sampled['sentiment'].values,
        test_size=0.2,
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test

class LSTMModel:
    def __init__(self, max_words=30000, max_len=300, embedding_dim=300):  # 增加词表大小和序列长度
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>", filters='')
        self.model = None
        self.model_dir = "models"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Add training history
        self.training_history = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.best_model_path = os.path.join(self.model_dir, f"lstm_best_weights_{self.timestamp}.keras")
        self.best_val_accuracy = 0
        self.best_val_metrics = {}
        self.best_epoch = 0
        
    def build_model(self):
        """Build an improved LSTM model architecture"""
        model = Sequential([
            # 增加词嵌入维度
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len, mask_zero=True),
            
            # 使用更深的LSTM层
            tf.keras.layers.Bidirectional(LSTM(256, return_sequences=True)),
            tf.keras.layers.BatchNormalization(),
            Dropout(0.3),
            
            tf.keras.layers.Bidirectional(LSTM(128, return_sequences=True)),
            tf.keras.layers.BatchNormalization(),
            Dropout(0.3),
            
            tf.keras.layers.Bidirectional(LSTM(64)),
            tf.keras.layers.BatchNormalization(),
            Dropout(0.3),
            
            # 增加全连接层
            Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            Dropout(0.3),
            
            # 输出层
            Dense(1, activation='sigmoid')
        ])
        
        # 使用Adam优化器，但降低学习率
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def preprocess_text(self, texts):
        """Enhanced text preprocessing for LSTM model"""
        # Convert all texts to strings and handle empty values
        texts = [str(text) if text is not None else '' for text in texts]
        
        # 简单的文本清理
        texts = [text.lower() for text in texts]  # 转小写
        texts = [re.sub(r'[^\w\s]', ' ', text) for text in texts]  # 移除标点符号
        texts = [re.sub(r'\s+', ' ', text).strip() for text in texts]  # 规范化空白字符
        
        # Fit tokenizer on texts
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding='post',
            truncating='post',
            value=0
        )
        
        return padded_sequences
    
    def save_model(self):
        """Save model and tokenizer"""
        if self.model is None:
            print("No model to save")
            return
            
        model_path = os.path.join(self.model_dir, "lstm_model.keras")
        tokenizer_path = os.path.join(self.model_dir, "lstm_tokenizer.pkl")
        history_path = os.path.join(self.model_dir, "lstm_history.json")
        
        # Save model
        self.model.save(model_path)
        
        # Save tokenizer
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
            
        # Save training history
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f)
            
        print(f"Model saved to: {model_path}")
        print(f"Tokenizer saved to: {tokenizer_path}")
        print(f"Training history saved to: {history_path}")
    
    def load_model(self):
        """Load saved model and tokenizer"""
        model_path = os.path.join(self.model_dir, "lstm_model.keras")
        tokenizer_path = os.path.join(self.model_dir, "lstm_tokenizer.pkl")
        history_path = os.path.join(self.model_dir, "lstm_history.json")
        
        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            print("No saved model found, will use new model")
            return False
            
        try:
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            
            # Load tokenizer
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
                
            # Load training history
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.training_history = json.load(f)
                    
            print(f"Model loaded from: {model_path}")
            print(f"Tokenizer loaded from: {tokenizer_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def train(self, X_train, y_train, epochs=20, batch_size=32, validation_split=0.2):  # 增加训练轮次，调整批量大小
        """Train the LSTM model with optimized parameters"""
        if self.model is None:
            self.build_model()
            
        # Preprocess text data
        X_train_seq = self.preprocess_text(X_train)
        
        # Create validation set
        val_size = int(len(X_train_seq) * validation_split)
        X_val = X_train_seq[-val_size:]
        y_val = y_train[-val_size:]
        X_train_seq = X_train_seq[:-val_size]
        y_train_seq = y_train[:-val_size]
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,  # 增加早停的耐心值
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.best_model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,  # 更激进的学习率衰减
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: self._update_best_model(epoch, logs)
            )
        ]
        
        # Calculate class weights
        class_weights = {
            0: len(y_train_seq) / (2 * np.sum(y_train_seq == 0)),
            1: len(y_train_seq) / (2 * np.sum(y_train_seq == 1))
        }
        
        # Train the model
        history = self.model.fit(
            X_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Save training history
        self.training_history = history.history
        
        # Load the best weights
        if os.path.exists(self.best_model_path):
            self.model = tf.keras.models.load_model(self.best_model_path)
            print(f"\nLoaded best model weights from epoch {self.best_epoch}")
            print(f"Best validation metrics:")
            print(f"  Accuracy:  {self.best_val_metrics['val_accuracy']:.4f}")
            print(f"  AUC:       {self.best_val_metrics['val_auc']:.4f}")
            print(f"  Precision: {self.best_val_metrics['val_precision']:.4f}")
            print(f"  Recall:    {self.best_val_metrics['val_recall']:.4f}")
        
        return history
    
    def _update_best_model(self, epoch, logs):
        """Update best model information"""
        if logs.get('val_accuracy', 0) > self.best_val_accuracy:
            self.best_val_accuracy = logs['val_accuracy']
            self.best_epoch = epoch + 1
            # Save all validation metrics
            self.best_val_metrics = {
                'val_accuracy': logs.get('val_accuracy', 0),
                'val_auc': logs.get('val_auc', 0),
                'val_precision': logs.get('val_precision', 0),
                'val_recall': logs.get('val_recall', 0),
                'val_loss': logs.get('val_loss', 0)
            }
            print(f"\nNew best model found at epoch {self.best_epoch}")
            print(f"Validation metrics:")
            print(f"  Accuracy:  {self.best_val_metrics['val_accuracy']:.4f}")
            print(f"  AUC:       {self.best_val_metrics['val_auc']:.4f}")
            print(f"  Precision: {self.best_val_metrics['val_precision']:.4f}")
            print(f"  Recall:    {self.best_val_metrics['val_recall']:.4f}")
            print(f"  Loss:      {self.best_val_metrics['val_loss']:.4f}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data using the best weights"""
        if self.model is None:
            print("No model to evaluate")
            return 0, "", []
            
        # Always load the best model weights for evaluation
        if os.path.exists(self.best_model_path):
            self.model = tf.keras.models.load_model(self.best_model_path)
            print(f"\nEvaluating using best model weights from epoch {self.best_epoch} (validation accuracy: {self.best_val_accuracy:.4f})")
        else:
            print("Warning: No best model weights found, using current model weights")
            
        # Preprocess text data
        X_test_seq = self.preprocess_text(X_test)
        
        # Get predictions
        y_pred_prob = self.model.predict(X_test_seq)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return accuracy, report, y_pred.tolist()

def compare_model_results(results_dict):
    """Compare and display results from different models"""
    print("\n" + "="*50)
    print("Model Performance Comparison")
    print("="*50)
    
    # Create comparison table
    print("\n{:<20} {:<15} {:<15} {:<15} {:<15}".format(
        "Model", "Val Accuracy", "Val AUC", "Val Precision", "Val Recall"))
    print("-"*80)
    
    for model_name, result in results_dict.items():
        if model_name == 'LSTM':
            # For LSTM, use the best validation metrics
            metrics = result.get('best_val_metrics', {})
            print("{:<20} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f}".format(
                model_name,
                metrics.get('val_accuracy', 0),
                metrics.get('val_auc', 0),
                metrics.get('val_precision', 0),
                metrics.get('val_recall', 0)
            ))
        else:
            # For other models, parse classification report
            report_lines = result['report'].split('\n')
            metrics = {'precision': 0, 'recall': 0}
            for line in report_lines:
                if '1' in line and 'avg' not in line and 'precision' not in line:
                    try:
                        parts = [x for x in line.split() if x]
                        if len(parts) >= 4:
                            metrics['precision'] = float(parts[1])
                            metrics['recall'] = float(parts[2])
                            break
                    except (ValueError, IndexError):
                        continue
            
            print("{:<20} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f}".format(
                model_name,
                result['accuracy'],
                result.get('auc', 0),
                metrics['precision'],
                metrics['recall']
            ))
    
    print("\n" + "="*50)
    
    # Save comparison results
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    comparison_file = os.path.join(results_dir, "model_comparison.json")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
    print(f"\nComparison results saved to: {comparison_file}")

def main():
    # Set data file path
    data_file = 'test.csv'
    sample_size = 5000  # 增加训练数据量
    
    try:
        # Load data
        print(f"\nLoading data file: {data_file}")
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            data_file, 
            sample_size=sample_size
        )
        print(f"Data loading complete. Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        print(f"Positive review ratio in training set: {sum(y_train == 1)/len(y_train):.2%}")
        print(f"Positive review ratio in test set: {sum(y_test == 1)/len(y_test):.2%}")
        
        # Dictionary to store all model results
        all_results = {}
        
        # 1. Logistic Regression model
        print("\nTraining Logistic Regression model...")
        lr_model = TraditionalModel(model_type='lr')
        lr_model.train(X_train, y_train, incremental=True)
        accuracy, report, predictions = lr_model.evaluate(X_test, y_test)
        print(f"Logistic Regression model accuracy: {accuracy:.4f}")
        print("Classification report:")
        print(report)
        save_results("logistic_regression", accuracy, report, predictions)
        all_results['Logistic Regression'] = {'accuracy': accuracy, 'report': report}
        
        # 2. Naive Bayes model
        print("\nTraining Naive Bayes model...")
        nb_model = TraditionalModel(model_type='nb')
        nb_model.train(X_train, y_train, incremental=True)
        accuracy, report, predictions = nb_model.evaluate(X_test, y_test)
        print(f"Naive Bayes model accuracy: {accuracy:.4f}")
        print("Classification report:")
        print(report)
        save_results("naive_bayes", accuracy, report, predictions)
        all_results['Naive Bayes'] = {'accuracy': accuracy, 'report': report}
        
        # 3. SVM model
        print("\nTraining SVM model...")
        svm_model = TraditionalModel(model_type='svm')
        svm_model.train(X_train, y_train, incremental=True)
        accuracy, report, predictions = svm_model.evaluate(X_test, y_test)
        print(f"SVM model accuracy: {accuracy:.4f}")
        print("Classification report:")
        print(report)
        save_results("svm", accuracy, report, predictions)
        all_results['SVM'] = {'accuracy': accuracy, 'report': report}
        
        # 4. LSTM model
        print("\nTraining LSTM model...")
        lstm_model = LSTMModel()
        lstm_model.train(X_train, y_train, epochs=20, batch_size=32)
        accuracy, report, predictions = lstm_model.evaluate(X_test, y_test)
        print(f"LSTM model validation accuracy: {lstm_model.best_val_accuracy:.4f}")
        print("Classification report:")
        print(report)
        save_results("lstm", accuracy, report, predictions)
        all_results['LSTM'] = {
            'accuracy': accuracy, 
            'report': report,
            'best_val_metrics': lstm_model.best_val_metrics  # Add best validation metrics
        }
        
        # Compare all model results
        compare_model_results(all_results)
        
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found")
        print("Please ensure the data file is in the correct directory")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 