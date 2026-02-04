# # import numpy as np
# # import cv2
# # import tensorflow as tf
# # from tensorflow import keras
# # import matplotlib.pyplot as plt
# # import matplotlib.cm as cm
# # from pathlib import Path

# # class HeatmapGenerator:
# #     """Simple heatmap generator"""
    
# #     def __init__(self, model):
# #         self.model = model
# #         print(f"🔍 Searching for conv layers in model...")
        
# #         # Find last conv layer
# #         self.conv_layer_name = None
# #         for layer in reversed(self.model.layers):
# #             layer_name = layer.name.lower()
# #             if 'conv' in layer_name:
# #                 self.conv_layer_name = layer.name
# #                 print(f"✅ Found conv layer: {layer.name}")
# #                 break
        
# #         if not self.conv_layer_name:
# #             print("⚠️  No conv layer found, will use simpler heatmap")
    
# #     def create_heatmap(self, image_path, output_path):
# #         """Generate heatmap"""
# #         try:
# #             print(f"🎨 Creating heatmap for: {image_path}")
            
# #             # Load and prepare image
# #             img = cv2.imread(image_path)
# #             if img is None:
# #                 raise ValueError(f"Cannot read image: {image_path}")
            
# #             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #             img_resized = cv2.resize(img_rgb, (380, 380))
# #             img_array = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
            
# #             # If we have a conv layer, use Grad-CAM
# #             if self.conv_layer_name:
# #                 heatmap = self._gradcam_heatmap(img_array)
# #             else:
# #                 # Fallback: simple attention map
# #                 heatmap = self._simple_heatmap(img_array)
            
# #             # Create visualization
# #             self._save_visualization(img_rgb, heatmap, output_path)
            
# #             print(f"✅ Heatmap saved to: {output_path}")
# #             return output_path
            
# #         except Exception as e:
# #             print(f"❌ Heatmap generation failed: {e}")
# #             import traceback
# #             traceback.print_exc()
# #             raise
    
# #     def _gradcam_heatmap(self, img_array):
# #         """Generate Grad-CAM heatmap"""
# #         try:
# #             # Create gradient model
# #             grad_model = keras.Model(
# #                 inputs=[self.model.input],
# #                 outputs=[
# #                     self.model.get_layer(self.conv_layer_name).output,
# #                     self.model.output
# #                 ]
# #             )
            
# #             with tf.GradientTape() as tape:
# #                 conv_outputs, predictions = grad_model(img_array)
# #                 pred_index = tf.argmax(predictions[0])
# #                 class_channel = predictions[:, pred_index]
            
# #             # Get gradients
# #             grads = tape.gradient(class_channel, conv_outputs)
            
# #             # Pool gradients
# #             pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
# #             # Weight channels
# #             conv_outputs = conv_outputs[0]
# #             heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
# #             heatmap = tf.squeeze(heatmap)
            
# #             # Normalize
# #             heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
# #             return heatmap.numpy()
            
# #         except Exception as e:
# #             print(f"⚠️  Grad-CAM failed: {e}, using simple heatmap")
# #             return self._simple_heatmap(img_array)
    
# #     def _simple_heatmap(self, img_array):
# #         """Simple attention heatmap (fallback)"""
# #         # Get prediction
# #         pred = self.model.predict(img_array, verbose=0)
# #         attention_strength = abs(pred[0][1] - pred[0][0])  # Difference between classes
        
# #         # Create center-focused heatmap (faces usually in center)
# #         size = 380
# #         y, x = np.ogrid[:size, :size]
# #         center_y, center_x = size // 2, size // 2
        
# #         # Gaussian-like distribution
# #         heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (size/3)**2))
# #         heatmap = heatmap * attention_strength
        
# #         return heatmap
    
# #     def _save_visualization(self, original_img, heatmap, output_path):
# #         """Save heatmap visualization"""
# #         # Resize heatmap to match original
# #         heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        
# #         # Apply colormap
# #         heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
# #         heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
# #         # Overlay
# #         alpha = 0.4
# #         superimposed = cv2.addWeighted(original_img, 1-alpha, heatmap_colored, alpha, 0)
        
# #         # Create figure with 3 subplots
# #         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
# #         axes[0].imshow(original_img)
# #         axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
# #         axes[0].axis('off')
        
# #         axes[1].imshow(heatmap, cmap='jet')
# #         axes[1].set_title('AI Attention Heatmap', fontsize=14, fontweight='bold')
# #         axes[1].axis('off')
        
# #         axes[2].imshow(superimposed)
# #         axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
# #         axes[2].axis('off')
        
# #         plt.tight_layout()
        
# #         # Ensure directory exists
# #         Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
# #         # Save
# #         plt.savefig(output_path, dpi=150, bbox_inches='tight')
# #         plt.close()
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.applications.efficientnet import preprocess_input
# import matplotlib.cm as cm
# from pathlib import Path


# class HeatmapGenerator:
#     """Grad-CAM for inference-only EfficientNet model"""

#     def __init__(self, model, img_size=240):
#         self.model = model
#         self.img_size = img_size
#         self.backbone = self.model.get_layer("efficientnetb0")
#         self.layer_name = "top_conv"

#     def create_heatmap(self, image_path, output_path):
#         img = cv2.imread(image_path)
#         if img is None:
#             raise ValueError("Image not found")

#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))

#         x = np.expand_dims(img_resized.astype("float32"), axis=0)
#         x = preprocess_input(x)

#         grad_model = keras.Model(
#             inputs=self.model.input,
#             outputs=[
#                 self.backbone.get_layer(self.layer_name).output,
#                 self.model.output
#             ],
#         )

#         with tf.GradientTape() as tape:
#             conv_out, preds = grad_model(x)
#             class_idx = tf.argmax(preds[0])
#             loss = preds[:, class_idx]

#         grads = tape.gradient(loss, conv_out)
#         pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#         conv_out = conv_out[0]
#         heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)

#         heatmap = tf.maximum(heatmap, 0)
#         heatmap /= tf.reduce_max(heatmap) + 1e-8
#         heatmap = heatmap.numpy()

#         heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#         heatmap_color = np.uint8(255 * cm.jet(heatmap)[:, :, :3])
#         overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)

#         Path(output_path).parent.mkdir(parents=True, exist_ok=True)
#         cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

#         return output_path

