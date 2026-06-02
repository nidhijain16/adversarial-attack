import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

# Ensure output directory for visualizations exists
os.makedirs("output_visualizations", exist_ok=True)

# 1. Define a simple PyTorch Audio Classifier (CNN over Mel-Spectrograms)
class AudioCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioCNN, self).__init__()
        # Input shape: (Batch, Channels=1, MelBins, TimeSteps)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # The linear layer input size depends on the exact audio length and mel bins.
        # This is a simplified architecture for demonstration.
        self.fc = nn.LazyLinear(num_classes) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 2. Digital Signal Processing (DSP): Spectrogram Visualization
def plot_spectrogram(waveform, sample_rate, title, filepath):
    """
    Computes and plots the Mel-spectrogram of an audio waveform.
    Demonstrates DSP skills using torchaudio and librosa.
    """
    # Convert waveform to Mel-spectrogram
    mel_spectrogram_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=64)
    mel_spec = mel_spectrogram_transform(waveform)
    
    # Plot using librosa
    plt.figure(figsize=(10, 4))
    mel_spec_db = librosa.power_to_db(mel_spec[0].detach().numpy(), ref=np.max)
    librosa.display.specshow(mel_spec_db, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Saved spectrogram visualization to {filepath}")

def main():
    print("--- Starting PyTorch Audio Adversarial Attack Pipeline ---")
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate a clean sine wave tone (440 Hz) for demonstration
    # Using a pure tone makes the adversarial perturbation highly visible on the spectrogram!
    sample_rate = 16000
    duration = 1.0 # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    clean_audio_np = 0.5 * np.sin(2 * np.pi * 440 * t) # 440 Hz sine wave
    dummy_waveform = torch.tensor(clean_audio_np, dtype=torch.float32).unsqueeze(0)
    
    # 3. Model Setup
    model = AudioCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # We need to wrap the PyTorch model for ART.
    # ART expects the input to be numpy arrays, but we will configure the input shape.
    # We will attack the raw waveform directly (1D), so the model should ideally 
    # handle the transform internally, but for simplicity we assume the model takes the spectrogram.
    # Let's write a wrapper model that takes raw waveform and converts it inside the forward pass, 
    # making it an end-to-end differentiable audio model.
    
    class EndToEndAudioModel(nn.Module):
        def __init__(self, base_model, sample_rate):
            super().__init__()
            self.mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=64)
            self.base_model = base_model
            
        def forward(self, x):
            # x is raw waveform (Batch, Time)
            # Add channel dimension: (Batch, 1, Mel, Time)
            x = self.mel_transform(x)
            x = x.unsqueeze(1) 
            return self.base_model(x)

    e2e_model = EndToEndAudioModel(model, sample_rate).to(device)
    
    # Dummy forward pass to initialize LazyLinear
    dummy_input = dummy_waveform.to(device)
    _ = e2e_model(dummy_input)

    # 4. Wrap with ART PyTorchClassifier
    classifier = PyTorchClassifier(
        model=e2e_model,
        clip_values=(-1.0, 1.0), # Audio signals are typically normalized between -1 and 1
        loss=criterion,
        optimizer=optimizer,
        input_shape=(int(sample_rate * duration),),
        nb_classes=10,
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )

    # 5. Define Adversarial Attack (FGSM)
    # We target the audio waveform itself, adding subtle noise.
    attack = FastGradientMethod(estimator=classifier, eps=0.05)

    # Generate adversarial audio
    print("Generating adversarial audio perturbation...")
    original_audio_np = dummy_input.cpu().numpy()
    adversarial_audio_np = attack.generate(x=original_audio_np)
    
    adversarial_audio = torch.tensor(adversarial_audio_np)
    original_audio = torch.tensor(original_audio_np)

    # 6. Visualize the impact of the attack (DSP Analysis)
    print("Generating DSP Visualizations (Mel-Spectrograms)...")
    plot_spectrogram(
        original_audio, 
        sample_rate, 
        "Original Audio Mel-Spectrogram", 
        "output_visualizations/original_spectrogram.png"
    )
    
    plot_spectrogram(
        adversarial_audio, 
        sample_rate, 
        "Adversarial Audio Mel-Spectrogram (FGSM)", 
        "output_visualizations/adversarial_spectrogram.png"
    )
    
    # Difference
    noise = adversarial_audio - original_audio
    plot_spectrogram(
        noise, 
        sample_rate, 
        "Adversarial Perturbation (Noise)", 
        "output_visualizations/perturbation_spectrogram.png"
    )

    print("Pipeline completed successfully. You can find the visualizations in 'output_visualizations/'.")

if __name__ == "__main__":
    main()
