"""
Vision-Language Model Inference Wrapper
Uses Qwen2.5-VL for fault diagnosis classification.
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os


class VLMClassifier(nn.Module):
    """VLM-based fault classifier using Qwen2.5-VL.

    Takes merged visual tokens from SGVT and classifies fault type.
    Uses LoRA for efficient fine-tuning.
    """

    def __init__(self, model_path, num_classes=4, lora_rank=16, lora_alpha=32,
                 device="cuda", use_lora=True):
        super().__init__()
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = device
        self.use_lora = use_lora

        # Load Qwen2.5-VL model
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Apply LoRA if specified
        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

        # Classification head on top of VLM hidden states
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward_from_image(self, images, labels=None):
        """Forward pass from spectrogram images.

        Args:
            images: (B, 3, H, W) spectrogram images tensor
            labels: (B,) optional class labels

        Returns:
            logits: (B, num_classes) classification logits
            loss: scalar loss (if labels provided)
        """
        B = images.shape[0]

        # Convert tensor images to PIL for processor
        pil_images = []
        for i in range(B):
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))

        # Create text prompt for classification
        class_names = ["Normal", "Ball fault", "Inner race fault", "Outer race fault"]
        if self.num_classes == 3:
            class_names = ["Baseline", "Outer race fault", "Inner race fault"]

        prompt = "Classify this vibration signal spectrogram. The fault type is:"

        # Process inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in pil_images
                ] + [{"type": "text", "text": prompt}],
            }
        ]

        # Use processor to prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text] * B,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Forward through VLM
        outputs = self.model(**inputs, output_hidden_states=True)
        # Use last hidden state of the last token for classification
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # (B, hidden_size)

        # Classification head
        logits = self.classifier(last_hidden)  # (B, num_classes)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return logits, loss

    def forward_from_tokens(self, merged_tokens, labels=None):
        """Forward pass from merged visual tokens.

        This is the main path for SGVT: merged tokens are injected
        into the VLM's visual processing pipeline.

        Args:
            merged_tokens: (B, K, D) merged visual tokens from SGVT
            labels: (B,) optional class labels

        Returns:
            logits: (B, num_classes)
            loss: scalar
        """
        B, K, D = merged_tokens.shape

        # Project tokens to VLM's visual embedding space
        # This requires interfacing with Qwen2.5-VL's visual encoder
        # For now, we use a simple projection
        if not hasattr(self, 'token_proj'):
            hidden_size = self.model.config.hidden_size
            self.token_proj = nn.Linear(D, hidden_size).to(merged_tokens.device)

        projected_tokens = self.token_proj(merged_tokens)  # (B, K, hidden_size)

        # Create a simple classification from merged tokens
        # Mean pooling over groups
        pooled = projected_tokens.mean(dim=1)  # (B, hidden_size)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return logits, loss


class SimpleVLMClassifier(nn.Module):
    """Simplified VLM classifier that doesn't require full VLM inference.

    Uses CLIP vision encoder + classification head.
    Much faster for training; full VLM used only for final evaluation.
    """

    def __init__(self, num_classes=4, feature_dim=768, device="cuda"):
        super().__init__()
        self.num_classes = num_classes
        self.device = device

        # Use CLIP vision encoder for feature extraction
        from transformers import CLIPVisionModel, CLIPImageProcessor
        self.clip_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch16"
        ).to(device)
        self.clip_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch16"
        )

        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Classification head
        clip_dim = self.clip_model.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Linear(clip_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, images, labels=None):
        """Forward pass using CLIP features.

        Args:
            images: (B, 3, H, W) spectrogram images
            labels: (B,) optional labels

        Returns:
            logits: (B, num_classes)
            loss: scalar
        """
        B = images.shape[0]

        # Convert to PIL for CLIP processor
        pil_images = []
        for i in range(B):
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))

        inputs = self.clip_processor(images=pil_images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            clip_outputs = self.clip_model(**inputs)
            # Use CLS token
            features = clip_outputs.last_hidden_state[:, 0, :]  # (B, 768)

        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fn(logits, labels)

        return logits, loss
