import os
import sys
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detection.settings')
import django
django.setup()

from django.conf import settings

# ============================================================================
# GPU CONFIGURATION - Optimized for RTX 3050 6GB
# ============================================================================
print("\n" + "=" * 70)
print("🎮 GPU CONFIGURATION (RTX 3050 6GB Optimization)")
print("=" * 70)

# Enable mixed precision training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print("✅ Mixed precision enabled (float16)")

# Configure GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=5632)]
        )
        print(f"✅ GPU detected: {len(gpus)} device(s)")
        print("✅ Memory limit: 5.5GB (safe for 6GB VRAM)")
    except RuntimeError as e:
        print(f"⚠️  GPU error: {e}")
else:
    print("❌ No GPU detected, using CPU")

# tf.config.optimizer.set_jit(True)
# print("✅ XLA compilation enabled")
# print("=" * 70 + "\n")


class DeepfakeTrainer:
    """High-accuracy trainer with proper phase separation"""
    
    def __init__(self):
        self.image_size = settings.IMAGE_SIZE
        self.batch_size = 16 
        self.epochs = 50
        self.model = None
        self.history = None
        self.threshold = settings.DETECTION_THRESHOLD
        
    def build_model(self):
        """Build model with proper architecture"""
        print("🏗️  Building Model...")
        
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(self.image_size, self.image_size, 3),
            pooling=None
        )
        
        # Freeze base initially
        base_model.trainable = False
        
        # Build model
        inputs = layers.Input(shape=(self.image_size, self.image_size, 3))
        
        # No augmentation in model (handled by ImageDataGenerator)
        x = base_model(inputs, training=False)
        
        # Simplified head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer with explicit float32
        outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name='predictions')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='deepfake_detector')
        
        print("✅ Model built!")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Threshold: {self.threshold*100}%")
        
        self.model = model
        self.base_model = base_model
        return model
    
    def prepare_data(self):
        """Prepare data with aggressive augmentation"""
        print("\n📂 Loading dataset...\n")
        
        train_dir = settings.DATASET['TRAIN']
        val_dir = settings.DATASET['VAL']
        
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found!\n"
                f"Train: {train_dir}\n"
                f"Val: {val_dir}\n"
            )
        
        # AGGRESSIVE augmentation
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1],
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['real', 'fake'],
            shuffle=True,
            seed=42
        )
        
        val_gen = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['real', 'fake'],
            shuffle=False
        )
        
        print("✅ Dataset loaded:")
        print(f"   Train: {train_gen.samples} images")
        print(f"   Val: {val_gen.samples} images")
        print(f"   Class mapping: {train_gen.class_indices}")
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(train_gen)
        print(f"   Class weights: {class_weights}")
        
        return train_gen, val_gen, class_weights
    
    def calculate_class_weights(self, generator):
        """Calculate class weights for imbalanced data"""
        counter = {0: 0, 1: 0}
        for labels in generator.labels:
            counter[int(labels)] += 1
        
        total = sum(counter.values())
        weight_for_0 = total / (2.0 * counter[0])
        weight_for_1 = total / (2.0 * counter[1])
        
        return {0: weight_for_0, 1: weight_for_1}
    
    def train(self):
        """Progressive training with proper phase separation"""
        print("\n" + "=" * 70)
        print("🚀 TRAINING START")
        print("=" * 70 + "\n")
        
        start_time = time.time()
        
        self.build_model()
        train_gen, val_gen, class_weights = self.prepare_data()
        
        model_dir = project_root / 'detector' / 'ml_models'
        model_dir.mkdir(parents=True, exist_ok=True)
        final_path = str(settings.MODEL_PATH)
        
        # ================================================================
        # PHASE 1: Train head only
        # ================================================================
        phase1_epochs = 25
        print("\n" + "=" * 70)
        print(f"📚 PHASE 1: Training head ({phase1_epochs} epochs)")
        print("=" * 70 + "\n")
        
        self.base_model.trainable = False
        
        # Compile for Phase 1
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        # Phase 1 callbacks
        callbacks_phase1 = [
            keras.callbacks.ModelCheckpoint(
                final_path,
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        history1 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=phase1_epochs,
            callbacks=callbacks_phase1,
            class_weight=class_weights,
            verbose=1
        )
        
        # ================================================================
        # PHASE 2: Fine-tune
        # ================================================================
        phase2_epochs = 25
        print("\n" + "=" * 70)
        print(f"🔓 PHASE 2: Fine-tuning ({phase2_epochs} epochs)")
        print("=" * 70 + "\n")
        
        # Unfreeze top layers
        self.base_model.trainable = True
        for layer in self.base_model.layers[:-30]:
            layer.trainable = False
        
        # CRITICAL FIX: Recompile with SAME metrics for Phase 2
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        # Phase 2 callbacks (NO PredictionMonitor to avoid metric conflicts)
        callbacks_phase2 = [
            keras.callbacks.ModelCheckpoint(
                final_path,
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        history2 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=phase1_epochs + phase2_epochs,
            initial_epoch=phase1_epochs,
            callbacks=callbacks_phase2,
            class_weight=class_weights,
            verbose=1
        )
        
        # Combine histories
        self.history = {}
        for key in history1.history.keys():
            if key in history2.history:
                self.history[key] = history1.history[key] + history2.history[key]
        
        # Load best model
        print("\n📥 Loading best model...")
        self.model = keras.models.load_model(final_path)
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Training time: {elapsed/60:.1f} minutes")
    
    def evaluate(self):
        """Evaluate on test set"""
        test_dir = settings.DATASET['TEST']
        
        if not test_dir.exists():
            print("\n⚠️  No test set found")
            return
        
        print("\n" + "=" * 70)
        print("📈 EVALUATING ON TEST SET")
        print("=" * 70 + "\n")
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['real', 'fake'],
            shuffle=False
        )
        
        results = self.model.evaluate(test_gen, verbose=1)
        
        print("\n📊 Test Results:")
        for i, metric_name in enumerate(self.model.metrics_names):
            print(f"   {metric_name}: {results[i]:.4f}")
        
        # Detailed analysis
        print("\n🔍 Prediction Analysis:")
        predictions = self.model.predict(test_gen)
        pred_classes = (predictions.flatten() >= self.threshold).astype(int)
        true_classes = test_gen.classes
        
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(true_classes, pred_classes)
        print("\n   Confusion Matrix:")
        print(f"   [[TN={cm[0,0]}  FP={cm[0,1]}]")
        print(f"    [FN={cm[1,0]}  TP={cm[1,1]}]]")
        
        print("\n   Classification Report:")
        print(classification_report(true_classes, pred_classes, 
                                   target_names=['real', 'fake'],
                                   digits=4))
    
    def plot_history(self):
        """Plot training curves"""
        if not self.history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history['accuracy'], 'b-', label='Train')
        axes[0, 0].plot(self.history['val_accuracy'], 'r-', label='Val')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history['loss'], 'b-', label='Train')
        axes[0, 1].plot(self.history['val_loss'], 'r-', label='Val')
        axes[0, 1].set_title('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC
        axes[1, 0].plot(self.history['auc'], 'b-', label='Train')
        axes[1, 0].plot(self.history['val_auc'], 'r-', label='Val')
        axes[1, 0].set_title('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Precision & Recall
        axes[1, 1].plot(self.history['precision'], 'b-', label='Precision')
        axes[1, 1].plot(self.history['recall'], 'g-', label='Recall')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        save_path = project_root / 'training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Training curves saved: {save_path}")
        plt.close()
    
    def summary(self):
        """Print summary"""
        best_acc = max(self.history.get('val_accuracy', [0]))
        best_auc = max(self.history.get('val_auc', [0]))
        
        print("\n" + "=" * 70)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 70)
        print(f"\n📈 Best validation accuracy: {best_acc*100:.2f}%")
        print(f"📈 Best validation AUC: {best_auc:.4f}")
        print(f"💾 Model saved: {settings.MODEL_PATH}")
        
        if best_acc > 0.90:
            print("\n   🌟 EXCELLENT performance!")
        elif best_acc > 0.75:
            print("\n   ✅ GOOD performance")
        elif best_acc > 0.60:
            print("\n   ⚠️  FAIR performance")
        else:
            print("\n   ❌ POOR performance - Check dataset!")
        
        print("\n💡 Next steps:")
        print("   1. Check training_history.png")
        print("   2. Test the model")
        print("   3. Run server: python manage.py runserver")
        print("=" * 70 + "\n")


def main():
    try:
        trainer = DeepfakeTrainer()
        trainer.train()
        trainer.evaluate()
        trainer.plot_history()
        trainer.summary()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()