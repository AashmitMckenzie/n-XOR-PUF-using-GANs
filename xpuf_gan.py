import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import json
import glob
from datetime import datetime
import pandas as pd
import re
import argparse
from tqdm import tqdm
import time

# Check for GPU availability and configure
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"Found {len(physical_devices)} GPU(s)")
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device}")
        except:
            print(f"Failed to set memory growth for {device}")
    print("Using GPU for training")
else:
    print("No GPU found, using CPU")

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class XPUF_GAN:
    def __init__(self, challenge_dim=64, response_dim=1, latent_dim=100, n_xor=4, 
                 learning_rate=0.0002, beta1=0.5, use_mixed_precision=True):
        """
        Initialize the XPUF GAN model
        
        Args:
            challenge_dim: Dimension of the PUF challenge vector
            response_dim: Dimension of the PUF response vector
            latent_dim: Dimension of the random noise input
            n_xor: Number of XOR gates in the PUF
            learning_rate: Learning rate for Adam optimizer
            beta1: Beta1 parameter for Adam optimizer
            use_mixed_precision: Whether to use mixed precision training
        """
        self.challenge_dim = challenge_dim
        self.response_dim = response_dim
        self.latent_dim = latent_dim
        self.n_xor = n_xor
        self.learning_rate = learning_rate
        self.beta1 = beta1
        
        # Enable mixed precision if requested
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled")
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
                                  optimizer=Adam(learning_rate, beta1),
                                  metrics=['accuracy'])
        
        # Build the generator
        self.generator = self.build_generator()
        
        # Build the combined model
        self.combined = self.build_gan()
        self.combined.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate, beta1))
        
        # Create directories for results
        os.makedirs(f'model_checkpoints/{n_xor}XOR', exist_ok=True)
        os.makedirs(f'results/{n_xor}XOR', exist_ok=True)
        os.makedirs(f'synthetic_dataset/{n_xor}XOR', exist_ok=True)
        
    def build_generator(self):
        """Build the Generator model for XPUF CRP synthesis"""
        # Define model inputs
        noise_input = Input(shape=(self.latent_dim,))
        challenge_input = Input(shape=(self.challenge_dim,))
        
        # Merge inputs
        merged = Concatenate()([noise_input, challenge_input])
        
        # Adjust network complexity based on n_xor
        neurons_base = 128 * self.n_xor
        
        # Dense layers with batch normalization and leaky ReLU
        x = Dense(neurons_base)(merged)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Dense(neurons_base * 2)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Dense(neurons_base * 4)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        # Output layer for response generation (tanh activation for -1 to 1 range)
        response_output = Dense(self.response_dim, activation='tanh')(x)
        
        # Define generator model
        model = Model([noise_input, challenge_input], response_output, name=f"generator_{self.n_xor}XOR")
        return model
    
    def build_discriminator(self):
        """Build a Discriminator model to distinguish real/fake XPUF CRPs"""
        # Inputs
        challenge_input = Input(shape=(self.challenge_dim,))
        response_input = Input(shape=(self.response_dim,))
        
        # Merge inputs
        merged = Concatenate()([challenge_input, response_input])
        
        # Adjust network complexity based on n_xor
        neurons_base = 128 * self.n_xor
        
        # Dense layers with dropout and leaky ReLU
        x = Dense(neurons_base * 4)(merged)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        
        x = Dense(neurons_base * 2)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        
        x = Dense(neurons_base)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        
        # Output layer - real or fake
        validity = Dense(1, activation='sigmoid')(x)
        
        # Define discriminator model
        model = Model([challenge_input, response_input], validity, name=f"discriminator_{self.n_xor}XOR")
        return model
    
    def build_gan(self):
        """Build the combined GAN model"""
        # Freeze discriminator weights during generator training
        self.discriminator.trainable = False
        
        # GAN input (noise + challenge)
        noise = Input(shape=(self.latent_dim,))
        challenge = Input(shape=(self.challenge_dim,))
        
        # Generate responses from noise and challenge
        generated_response = self.generator([noise, challenge])
        
        # Discriminator determines validity of the generated responses
        validity = self.discriminator([challenge, generated_response])
        
        # Combined model trains generator to fool discriminator
        model = Model([noise, challenge], validity, name=f"gan_{self.n_xor}XOR")
        return model
    
    def train(self, real_challenges, real_responses, epochs=10000, batch_size=128, 
              save_interval=1000, validation_interval=500, checkpoint_dir=None):
        """
        Train the GAN model
        
        Args:
            real_challenges: Array of real PUF challenges
            real_responses: Array of real PUF responses
            epochs: Number of training epochs
            batch_size: Training batch size
            save_interval: Interval at which to save model checkpoints
            validation_interval: Interval at which to validate the model
            checkpoint_dir: Directory to save checkpoints
        """
        # Create directory for checkpoints if it doesn't exist
        if checkpoint_dir is None:
            checkpoint_dir = f'model_checkpoints/{self.n_xor}XOR'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Normalize responses to [-1, 1] for tanh activation
        real_responses_normalized = real_responses * 2 - 1
        
        # Training history
        d_losses = []
        g_losses = []
        d_accuracies = []
        
        # Define results directory
        results_dir = f'results/{self.n_xor}XOR'
        os.makedirs(results_dir, exist_ok=True)
        
        # Define adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        # Create progress bar
        progress_bar = tqdm(range(epochs), desc=f"Training {self.n_xor}XOR PUF GAN")
        
        # Training start time
        start_time = time.time()
        
        for epoch in progress_bar:
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select a random batch of real CRPs
            idx = np.random.randint(0, real_challenges.shape[0], batch_size)
            batch_challenges = real_challenges[idx]
            batch_responses = real_responses_normalized[idx]
            
            # Sample noise and generate a batch of new responses
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_responses = self.generator.predict([noise, batch_challenges], verbose=0)
            
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([batch_challenges, batch_responses], valid)
            d_loss_fake = self.discriminator.train_on_batch([batch_challenges, gen_responses], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            # Sample noise and generate new CRPs
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch([noise, batch_challenges], valid)
            
            # Store losses and accuracy for plotting
            d_losses.append(d_loss[0])
            d_accuracies.append(d_loss[1] * 100)
            g_losses.append(g_loss)
            
            # Update progress bar
            progress_bar.set_postfix({
                'D Loss': f"{d_loss[0]:.4f}", 
                'D Acc': f"{d_loss[1]*100:.2f}%", 
                'G Loss': f"{g_loss:.4f}"
            })
            
            # Periodically save model checkpoints
            if epoch % save_interval == 0:
                self.save_model(epoch, checkpoint_dir)
                
                # Plot training progress
                self.plot_training_progress(d_losses, g_losses, d_accuracies, epoch, results_dir)
            
            # Validate model periodically
            if epoch % validation_interval == 0 and epoch > 0:
                self.validate_model(real_challenges, real_responses_normalized, epoch, results_dir)
        
        # Calculate training time
        training_time = time.time() - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Save final model
        self.save_model('final', checkpoint_dir)
        
        # Final validation
        self.validate_model(real_challenges, real_responses_normalized, 'final', results_dir)
                
        return d_losses, g_losses, d_accuracies
    
    def save_model(self, epoch, checkpoint_dir=None):
        """Save model checkpoints"""
        if checkpoint_dir is None:
            checkpoint_dir = f'model_checkpoints/{self.n_xor}XOR'
        
        self.generator.save(f'{checkpoint_dir}/generator_epoch_{epoch}.h5')
        self.discriminator.save(f'{checkpoint_dir}/discriminator_epoch_{epoch}.h5')
    
    def load_model(self, epoch, checkpoint_dir=None):
        """Load model from checkpoints"""
        if checkpoint_dir is None:
            checkpoint_dir = f'model_checkpoints/{self.n_xor}XOR'
        
        self.generator = load_model(f'{checkpoint_dir}/generator_epoch_{epoch}.h5')
        self.discriminator = load_model(f'{checkpoint_dir}/discriminator_epoch_{epoch}.h5')
        
        # Rebuild combined model
        self.combined = self.build_gan()
        self.combined.compile(loss='binary_crossentropy', optimizer=Adam(self.learning_rate, self.beta1))
    
    def plot_training_progress(self, d_losses, g_losses, d_accuracies, epoch, results_dir=None):
        """Plot and save training progress"""
        if results_dir is None:
            results_dir = f'results/{self.n_xor}XOR'
        
        plt.figure(figsize=(15, 5))
        
        # Plot discriminator and generator loss
        plt.subplot(1, 2, 1)
        plt.plot(d_losses, label='Discriminator')
        plt.plot(g_losses, label='Generator')
        plt.title(f'{self.n_xor}XOR PUF - Model Losses')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot discriminator accuracy
        plt.subplot(1, 2, 2)
        plt.plot(d_accuracies)
        plt.title(f'{self.n_xor}XOR PUF - Discriminator Accuracy')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Epoch')
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/training_progress_epoch_{epoch}.png')
        plt.close()
    
    def validate_model(self, real_challenges, real_responses, epoch, results_dir=None):
        """Validate the model by comparing synthetic and real responses"""
        if results_dir is None:
            results_dir = f'results/{self.n_xor}XOR'
        
        # Generate synthetic responses for validation
        num_validation = min(1000, len(real_challenges))
        validation_challenges = real_challenges[:num_validation]
        noise = np.random.normal(0, 1, (num_validation, self.latent_dim))
        
        synthetic_responses = self.generator.predict([noise, validation_challenges], verbose=0)
        # Convert back from [-1, 1] to [0, 1]
        synthetic_responses_binary = (synthetic_responses > 0).astype(int)
        real_responses_binary = (real_responses > 0).astype(int)
        
        # Calculate validation metrics
        validation_metrics = self.calculate_validation_metrics(
            real_responses_binary[:num_validation], synthetic_responses_binary)
        
        # Save validation metrics
        with open(f'{results_dir}/validation_metrics_epoch_{epoch}.json', 'w') as f:
            json.dump(validation_metrics, f)
            
        print(f"\nEpoch {epoch} - Validation Metrics for {self.n_xor}XOR PUF:")
        print(f"  Hamming Weight - Real: {validation_metrics['hamming_weight_real']:.4f}, " +
              f"Synthetic: {validation_metrics['hamming_weight_synthetic']:.4f}")
        print(f"  Uniqueness - Real: {validation_metrics['uniqueness_real']:.2f}%, " +
              f"Synthetic: {validation_metrics['uniqueness_synthetic']:.2f}%")
        print(f"  Predictability: {validation_metrics['predictability']:.4f}")
    
    def calculate_validation_metrics(self, real_responses, synthetic_responses):
        """Calculate validation metrics for real and synthetic responses"""
        # Hamming weight (percentage of 1s)
        real_hw = np.mean(real_responses)
        synthetic_hw = np.mean(synthetic_responses)
        
        # Calculate inter-response Hamming distance for uniqueness
        def calculate_uniqueness(responses):
            num_samples = min(100, responses.shape[0])  # Limit computation for large datasets
            responses = responses[:num_samples]
            total_hd = 0
            count = 0
            
            for i in range(num_samples):
                for j in range(i+1, num_samples):
                    hd = np.sum(responses[i] != responses[j])
                    total_hd += hd
                    count += 1
            
            # Average hamming distance as percentage of response length
            if count > 0:
                return (total_hd / count) / responses.shape[1] * 100
            return 0
        
        real_uniqueness = calculate_uniqueness(real_responses)
        synthetic_uniqueness = calculate_uniqueness(synthetic_responses)
        
        # Predictability - measure how well synthetic data models real data distribution
        # Use a sample of real and synthetic responses for evaluation
        sample_size = min(200, len(real_responses), len(synthetic_responses))
        real_sample = real_responses[:sample_size].flatten()
        synthetic_sample = synthetic_responses[:sample_size].flatten()
        
        # Calculate predictability (1 - absolute difference in distributions)
        predictability = 1 - abs(np.mean(real_sample) - np.mean(synthetic_sample))
        
        return {
            'hamming_weight_real': float(real_hw),
            'hamming_weight_synthetic': float(synthetic_hw),
            'uniqueness_real': float(real_uniqueness),
            'uniqueness_synthetic': float(synthetic_uniqueness),
            'predictability': float(predictability)
        }
    
    def generate_synthetic_dataset(self, num_samples, output_format='binary', output_dir=None):
        """
        Generate a complete synthetic XPUF dataset
        
        Args:
            num_samples: Number of synthetic samples to generate
            output_format: Format of challenges in output ('binary' or 'hex')
            output_dir: Directory to save the dataset
            
        Returns:
            Tuple of (challenges, responses, metadata)
        """
        if output_dir is None:
            output_dir = f'synthetic_dataset/{self.n_xor}XOR'
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate random challenges
        synthetic_challenges = np.random.randint(0, 2, (num_samples, self.challenge_dim))
        
        # Generate responses in batches to avoid memory issues
        batch_size = 1000
        synthetic_responses_binary = np.zeros((num_samples, self.response_dim), dtype=int)
        
        print(f"Generating {num_samples} synthetic responses for {self.n_xor}XOR PUF...")
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_challenges = synthetic_challenges[i:batch_end]
            batch_size_actual = batch_end - i
            
            # Generate random latent space vectors
            latent_vectors = np.random.normal(0, 1, (batch_size_actual, self.latent_dim))
            
            # Generate responses
            batch_responses = self.generator.predict([latent_vectors, batch_challenges], verbose=0)
            
            # Convert from [-1, 1] to binary [0, 1]
            synthetic_responses_binary[i:batch_end] = (batch_responses > 0).astype(int)
        
        # Create metadata
        metadata = {
            'date_generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'n_xor': self.n_xor,
            'num_samples': num_samples,
            'challenge_dim': self.challenge_dim,
            'response_dim': self.response_dim,
            'validation_metrics': self.calculate_validation_metrics(
                synthetic_responses_binary[:100], synthetic_responses_binary[-100:])
        }
        
        # Save in binary format
        np.save(f'{output_dir}/synthetic_xpuf_challenges.npy', synthetic_challenges)
        np.save(f'{output_dir}/synthetic_xpuf_responses.npy', synthetic_responses_binary)
        
        # Also save in the same format as the original dataset
        with open(f'{output_dir}/synthetic_xpuf_crps.txt', 'w') as f:
            for i in range(num_samples):
                # Convert challenge to hex if requested
                if output_format == 'hex':
                    # Convert bit array to integer
                    challenge_int = int(''.join(map(str, synthetic_challenges[i])), 2)
                    # Convert integer to hex string (without 0x prefix)
                    challenge_hex = format(challenge_int, f'0{self.challenge_dim//4}X')
                    f.write(f"{challenge_hex};{synthetic_responses_binary[i][0]}\n")
                else:
                    # Binary format as string
                    challenge_str = ''.join(map(str, synthetic_challenges[i]))
                    f.write(f"{challenge_str};{synthetic_responses_binary[i][0]}\n")
        
        with open(f'{output_dir}/synthetic_xpuf_metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        print(f"Generated {num_samples} synthetic {self.n_xor}XOR PUF challenge-response pairs")
        print(f"Dataset saved in '{output_dir}' directory")
        
        return synthetic_challenges, synthetic_responses_binary, metadata
    
    def test_ml_attack_resistance(self, real_challenges, real_responses, results_dir=None):
        """Test ML attack resistance by training a model to predict responses"""
        if results_dir is None:
            results_dir = f'results/{self.n_xor}XOR'
            
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate synthetic dataset
        num_samples = min(real_challenges.shape[0], 10000)  # Limit size for faster evaluation
        synthetic_challenges, synthetic_responses, _ = self.generate_synthetic_dataset(num_samples)
        
        # Build simple ML attack model with parallel processing
        def build_ml_attack_model(input_dim, output_dim):
            # Use strategy for distributed training if multiple GPUs available
            if len(physical_devices) > 1:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    model = tf.keras.Sequential([
                        Dense(512, input_dim=input_dim, activation='relu'),
                        Dropout(0.3),
                        Dense(256, activation='relu'),
                        Dropout(0.3),
                        Dense(output_dim, activation='sigmoid')
                    ])
                    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            else:
                model = tf.keras.Sequential([
                    Dense(512, input_dim=input_dim, activation='relu'),
                    Dropout(0.3),
                    Dense(256, activation='relu'),
                    Dropout(0.3),
                    Dense(output_dim, activation='sigmoid')
                ])
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model
        
        # Split datasets
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            real_challenges, real_responses, test_size=0.3)
        
        X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(
            synthetic_challenges, synthetic_responses, test_size=0.3)
        
        # Build models
        print(f"\nTraining ML attack models for {self.n_xor}XOR PUF...")
        model_real = build_ml_attack_model(X_train_real.shape[1], y_train_real.shape[1])
        model_syn = build_ml_attack_model(X_train_syn.shape[1], y_train_syn.shape[1])
        
        # Train on real data
        print("Training model on real data...")
        model_real.fit(
            X_train_real, y_train_real, 
            epochs=50, batch_size=64, 
            verbose=1,
            validation_data=(X_test_real, y_test_real)
        )
        
        # Train on synthetic data
        print("Training model on synthetic data...")
        model_syn.fit(
            X_train_syn, y_train_syn, 
            epochs=50, batch_size=64, 
            verbose=1,
            validation_data=(X_test_syn, y_test_syn)
        )
        
        # Evaluate both models on real test data
        print("Evaluating models...")
        real_on_real = model_real.evaluate(X_test_real, y_test_real)[1]  # accuracy
        syn_on_real = model_syn.evaluate(X_test_real, y_test_real)[1]  # accuracy
        
        # Evaluate on synthetic test data
        real_on_syn = model_real.evaluate(X_test_syn, y_test_syn)[1]  # accuracy
        syn_on_syn = model_syn.evaluate(X_test_syn, y_test_syn)[1]  # accuracy
        
        results = {
            'n_xor': self.n_xor,
            'real_model_on_real_data': float(real_on_real),
            'synthetic_model_on_real_data': float(syn_on_real),
            'real_model_on_synthetic_data': float(real_on_syn),
            'synthetic_model_on_synthetic_data': float(syn_on_syn),
            'difference_on_real_data': float(abs(real_on_real - syn_on_real))
        }
        
        # Save results
        with open(f'{results_dir}/ml_attack_results.json', 'w') as f:
            json.dump(results, f)
        
        print("\nML Attack Success Rate:")
        print(f"  Model trained on real data, tested on real data: {real_on_real:.4f}")
        print(f"  Model trained on synthetic data, tested on real data: {syn_on_real:.4f}")
        print(f"  Model trained on real data, tested on synthetic data: {real_on_syn:.4f}")
        print(f"  Model trained on synthetic data, tested on synthetic data: {syn_on_syn:.4f}")
        print(f"  Difference on real data: {abs(real_on_real - syn_on_real):.4f}")
        
        return results

    def reliability_analysis(self, challenges, noise_levels=[0.01, 0.05, 0.1], results_dir=None):
        """Analyze reliability of synthetic PUF responses under noise"""
        if results_dir is None:
            results_dir = f'results/{self.n_xor}XOR'
        
        os.makedirs(results_dir, exist_ok=True)
        
        num_samples = min(1000, challenges.shape[0])
        challenges = challenges[:num_samples]
        
        # Generate baseline responses
        baseline_latent = np.random.normal(0, 1, (num_samples, self.latent_dim))
        baseline_responses = self.generator.predict([baseline_latent, challenges], verbose=0)
        baseline_binary = (baseline_responses > 0).astype(int)
        
        reliability_scores = []
        
        print(f"\nPerforming reliability analysis for {self.n_xor}XOR PUF...")
        for noise_level in noise_levels:
            bit_errors = []
            
            # Generate multiple noisy responses for each challenge
            for _ in range(10):
                # Add noise to the latent space
                noisy_latent = baseline_latent.copy()
                noisy_latent += np.random.normal(0, noise_level, noisy_latent.shape)
                
                # Generate responses with noisy latent
                noisy_resp = self.generator.predict([noisy_latent, challenges], verbose=0)
                noisy_binary = (noisy_resp > 0).astype(int)
                
                # Calculate bit error rate
                bit_error = np.mean(np.abs(baseline_binary - noisy_binary))
                bit_errors.append(bit_error)
            
            avg_bit_error = np.mean(bit_errors)
            reliability = 1 - avg_bit_error
            reliability_scores.append(reliability)
            
            print(f"Noise level {noise_level}: Reliability = {reliability*100:.2f}%")
        
        # Save reliability results
        reliability_results = {
            'n_xor': self.n_xor,
            'noise_levels': noise_levels,
            'reliability_scores': reliability_scores
        }
        
        with open(f'{results_dir}/reliability_results.json', 'w') as f:
            json.dump(reliability_results, f)
        
        return reliability_scores


def load_xpuf_dataset(filepath):
    """
    Load and preprocess the XPUF dataset
    
    Args:
        filepath: Path to the XPUF dataset file
        
    Returns:
        Tuple of (challenges, responses) as numpy arrays
    """
    # Read the file
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    challenges = []
    responses = []
    
    # Extract challenge-response pairs
    for line in lines:
        if ';' in line:
            parts = line.strip().split(';')
            if len(parts) == 2:
                challenge_str = parts[0]
                response = int(parts[1])
                
                # Check if challenge is hex or binary
                if all(c in '01' for c in challenge_str):
                    # Binary format
                    # Ensure correct length
                    challenge_dim = len(challenge_str)
                    bin_challenge = challenge_str.zfill(challenge_dim)
                else:
                    # Hex format
                    # Convert hex to binary
                    bin_challenge = bin(int(challenge_str, 16))[2:]
                    
                    # Pad to ensure correct length
                    challenge_dim = len(challenge_str) * 4  # Each hex digit is 4 bits
                    bin_challenge = bin_challenge.zfill(challenge_dim)
                
                # Reverse the order as specified in the original code
                bin_challenge = bin_challenge[::-1]
                
                # Convert to list of integers
                bin_challenge_list = [int(bit) for bit in bin_challenge]
                
                challenges.append(bin_challenge_list)
                responses.append([response])
    
    # Convert to numpy arrays
    challenges_array = np.array(challenges)
    responses_array = np.array(responses)
    
    print(f"Loaded {len(challenges)} challenge-response pairs")
    print(f"Challenge dimension: {challenges_array.shape[1]}")
    
    return challenges_array, responses_array


def extract_n_from_filename(filename):
    """Extract the n value from the XOR PUF filename"""
    match = re.search(r'(\d+)XOR', filename)
    if match:
        return int(match.group(1))
    return None

def train_all_models(base_dir, epochs=5000, batch_size=128):
    """Train GAN models for all XOR PUF configurations"""
    # Find all XOR PUF data files
    data_files = glob.glob(f"{base_dir}/*XOR_*_LUT_*.txt")
    
    if not data_files:
        print(f"No XOR PUF data files found in {base_dir}")
        return
    
    results_summary = []
    
    for data_file in sorted(data_files):
        # Extract n value from filename
        filename = os.path.basename(data_file)
        n = extract_n_from_filename(filename)
        
        if n is None:
            print(f"Could not detect n value for {filename}, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {n}-XOR PUF dataset: {filename}")
        print(f"{'='*60}")
        
        # Load the dataset
        challenges, responses = load_xpuf_dataset(data_file)
        
        # Get challenge dimension from the data
        challenge_dim = challenges.shape[1]
        response_dim = responses.shape[1]
        
        # Initialize and train the GAN
        xpuf_gan = XPUF_GAN(
            challenge_dim=challenge_dim, 
            response_dim=response_dim, 
            n_xor=n,
            use_mixed_precision=(len(physical_devices) > 0)  # Use mixed precision on GPU
        )
        
        # Train the GAN
        d_losses, g_losses, d_accuracies = xpuf_gan.train(
            challenges, responses, 
            epochs=epochs, 
            batch_size=batch_size, 
            save_interval=1000, 
            validation_interval=500
        )
        
        # Generate synthetic dataset - generate same number of samples as original
        num_samples = min(len(challenges), 100000)  # Cap at 100k for very large datasets
        synthetic_challenges, synthetic_responses, metadata = xpuf_gan.generate_synthetic_dataset(
            num_samples, output_format='hex')
        
        # Test ML attack resistance
        ml_results = xpuf_gan.test_ml_attack_resistance(challenges, responses)
        
        # Reliability analysis
        reliability_scores = xpuf_gan.reliability_analysis(challenges[:1000])
        
        # Add to results summary
        results_summary.append({
            'n_xor': n,
            'filename': filename,
            'samples': len(challenges),
            'ml_attack_resistance': ml_results['difference_on_real_data'],
            'predictability': metadata['validation_metrics']['predictability'],
            'reliability': np.mean(reliability_scores)
        })
    
    # Save summary results
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv('xpuf_gan_results_summary.csv', index=False)
    
    # Plot summary comparison
    plot_comparison_results(results_summary)
    
    return results_summary


def plot_comparison_results(results_summary):
    """Plot comparison of results across different XOR PUF configurations"""
    plt.figure(figsize=(15, 10))
    
    # Sort by n_xor
    results = sorted(results_summary, key=lambda x: x['n_xor'])
    n_xor_values = [r['n_xor'] for r in results]
    
    # Plot ML attack resistance
    plt.subplot(2, 2, 1)
    ml_resistance = [r['ml_attack_resistance'] for r in results]
    plt.bar(n_xor_values, ml_resistance)
    plt.title('ML Attack Resistance')
    plt.xlabel('XOR PUF Complexity (n)')
    plt.ylabel('Resistance Score')
    plt.xticks(n_xor_values)
    
    # Plot predictability
    plt.subplot(2, 2, 2)
    predictability = [r['predictability'] for r in results]
    plt.bar(n_xor_values, predictability)
    plt.title('Model Predictability')
    plt.xlabel('XOR PUF Complexity (n)')
    plt.ylabel('Predictability Score')
    plt.xticks(n_xor_values)
    
    # Plot reliability
    plt.subplot(2, 2, 3)
    reliability = [r['reliability'] for r in results]
    plt.bar(n_xor_values, reliability)
    plt.title('Reliability')
    plt.xlabel('XOR PUF Complexity (n)')
    plt.ylabel('Avg. Reliability Score')
    plt.xticks(n_xor_values)
    
    # Plot dataset size
    plt.subplot(2, 2, 4)
    samples = [r['samples'] for r in results]
    plt.bar(n_xor_values, samples)
    plt.title('Dataset Size')
    plt.xlabel('XOR PUF Complexity (n)')
    plt.ylabel('Number of Samples')
    plt.xticks(n_xor_values)
    
    plt.tight_layout()
    plt.savefig('xpuf_gan_comparison.png')
    plt.close()


def compare_configurations(base_dir, num_samples=10000):
    """Generate and compare synthetic data from all trained models"""
    # Find all model directories
    model_dirs = glob.glob(f"model_checkpoints/*XOR")
    
    if not model_dirs:
        print("No trained models found. Please train models first.")
        return
    
    comparison_results = []
    
    for model_dir in sorted(model_dirs):
        # Extract n value
        n = int(os.path.basename(model_dir).replace("XOR", ""))
        
        print(f"\nGenerating comparison data for {n}-XOR PUF...")
        
        # Find the latest model checkpoint
        generator_files = glob.glob(f"{model_dir}/generator_epoch_*.h5")
        if not generator_files:
            print(f"No generator model found in {model_dir}, skipping")
            continue
        
        # Get the latest checkpoint
        latest_checkpoint = max(generator_files, key=lambda x: 
                               'final' in x if 'final' in x else 
                               int(re.search(r'epoch_(\d+)', x).group(1)))
        
        epoch = 'final' if 'final' in latest_checkpoint else re.search(r'epoch_(\d+)', latest_checkpoint).group(1)
        print(f"Using checkpoint from epoch {epoch}")
        
        # Load the original dataset to get challenge dimension
        data_files = glob.glob(f"{base_dir}/{n}XOR_*_LUT_*.txt")
        if not data_files:
            print(f"No dataset found for {n}-XOR PUF, skipping")
            continue
        
        # Load a sample of challenges to get dimensions
        challenges, _ = load_xpuf_dataset(data_files[0])
        challenge_dim = challenges.shape[1]
        
        # Initialize the model
        xpuf_gan = XPUF_GAN(challenge_dim=challenge_dim, response_dim=1, n_xor=n)
        
        # Load the trained model
        xpuf_gan.load_model(epoch, checkpoint_dir=model_dir)
        
        # Generate synthetic data
        test_challenges = np.random.randint(0, 2, (num_samples, challenge_dim))
        
        # Generate responses
        noise = np.random.normal(0, 1, (num_samples, xpuf_gan.latent_dim))
        synthetic_responses = xpuf_gan.generator.predict([noise, test_challenges], verbose=0)
        synthetic_responses_binary = (synthetic_responses > 0).astype(int)
        
        # Calculate metrics
        uniqueness = xpuf_gan.calculate_validation_metrics(
            synthetic_responses_binary[:100], synthetic_responses_binary[-100:])['uniqueness_synthetic']
        
        hamming_weight = np.mean(synthetic_responses_binary)
        
        # Test reliability
        reliability_scores = xpuf_gan.reliability_analysis(test_challenges[:1000])
        avg_reliability = np.mean(reliability_scores)
        
        comparison_results.append({
            'n_xor': n,
            'uniqueness': uniqueness,
            'hamming_weight': float(hamming_weight),
            'reliability': avg_reliability
        })
    
    # Save comparison results
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv('xpuf_gan_model_comparison.csv', index=False)
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    # Sort by n_xor
    comparison_results = sorted(comparison_results, key=lambda x: x['n_xor'])
    n_xor_values = [r['n_xor'] for r in comparison_results]
    
    # Plot uniqueness
    plt.subplot(1, 3, 1)
    uniqueness = [r['uniqueness'] for r in comparison_results]
    plt.bar(n_xor_values, uniqueness)
    plt.title('Uniqueness')
    plt.xlabel('XOR PUF Complexity (n)')
    plt.ylabel('Uniqueness (%)')
    plt.xticks(n_xor_values)
    
    # Plot hamming weight
    plt.subplot(1, 3, 2)
    hamming_weight = [r['hamming_weight'] for r in comparison_results]
    plt.bar(n_xor_values, hamming_weight)
    plt.title('Hamming Weight')
    plt.xlabel('XOR PUF Complexity (n)')
    plt.ylabel('Average Hamming Weight')
    plt.xticks(n_xor_values)
    plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.5)  # Ideal is 0.5
    
    # Plot reliability
    plt.subplot(1, 3, 3)
    reliability = [r['reliability'] for r in comparison_results]
    plt.bar(n_xor_values, reliability)
    plt.title('Reliability')
    plt.xlabel('XOR PUF Complexity (n)')
    plt.ylabel('Average Reliability')
    plt.xticks(n_xor_values)
    
    plt.tight_layout()
    plt.savefig('xpuf_gan_model_comparison.png')
    plt.close()
    
    return comparison_results


def benchmark_generation_performance(base_dir):
    """Benchmark generation performance across different XOR PUF configurations"""
    # Find all model directories
    model_dirs = glob.glob(f"model_checkpoints/*XOR")
    
    if not model_dirs:
        print("No trained models found. Please train models first.")
        return
    
    benchmark_results = []
    
    for model_dir in sorted(model_dirs):
        # Extract n value
        n = int(os.path.basename(model_dir).replace("XOR", ""))
        
        print(f"\nBenchmarking generation performance for {n}-XOR PUF...")
        
        # Find the latest model
        generator_files = glob.glob(f"{model_dir}/generator_epoch_*.h5")
        if not generator_files:
            print(f"No generator model found in {model_dir}, skipping")
            continue
        
        # Get the latest checkpoint
        latest_checkpoint = max(generator_files, key=lambda x: 
                               'final' in x if 'final' in x else 
                               int(re.search(r'epoch_(\d+)', x).group(1)))
        
        epoch = 'final' if 'final' in latest_checkpoint else re.search(r'epoch_(\d+)', latest_checkpoint).group(1)
        
        # Load the original dataset to get challenge dimension
        data_files = glob.glob(f"{base_dir}/{n}XOR_*_LUT_*.txt")
        if not data_files:
            print(f"No dataset found for {n}-XOR PUF, skipping")
            continue
        
        # Load a sample of challenges to get dimensions
        challenges, _ = load_xpuf_dataset(data_files[0])
        challenge_dim = challenges.shape[1]
        
        # Initialize the model
        xpuf_gan = XPUF_GAN(challenge_dim=challenge_dim, response_dim=1, n_xor=n)
        
        # Load the trained model
        xpuf_gan.load_model(epoch, checkpoint_dir=model_dir)
        
        # Benchmark generation speed for different batch sizes
        batch_sizes = [100, 1000, 10000]
        times = []
        
        for batch_size in batch_sizes:
            # Prepare inputs
            test_challenges = np.random.randint(0, 2, (batch_size, challenge_dim))
            noise = np.random.normal(0, 1, (batch_size, xpuf_gan.latent_dim))
            
            # Warm-up run
            _ = xpuf_gan.generator.predict([noise[:10], test_challenges[:10]], verbose=0)
            
            # Timed run
            start_time = time.time()
            _ = xpuf_gan.generator.predict([noise, test_challenges], verbose=0)
            elapsed_time = time.time() - start_time
            
            times.append(elapsed_time)
            print(f"  Batch size {batch_size}: {elapsed_time:.4f} seconds " + 
                  f"({batch_size/elapsed_time:.2f} samples/second)")
        
        benchmark_results.append({
            'n_xor': n,
            'batch_size_100': times[0],
            'batch_size_1000': times[1],
            'batch_size_10000': times[2],
            'samples_per_second_100': 100/times[0],
            'samples_per_second_1000': 1000/times[1],
            'samples_per_second_10000': 10000/times[2]
        })
    
    # Save benchmark results
    benchmark_df = pd.DataFrame(benchmark_results)
    benchmark_df.to_csv('xpuf_gan_generation_benchmark.csv', index=False)
    
    # Plot benchmark results
    plt.figure(figsize=(12, 6))
    
    # Sort by n_xor
    benchmark_results = sorted(benchmark_results, key=lambda x: x['n_xor'])
    n_xor_values = [r['n_xor'] for r in benchmark_results]
    
    # Plot samples per second for different batch sizes
    plt.subplot(1, 2, 1)
    s100 = [r['samples_per_second_100'] for r in benchmark_results]
    s1000 = [r['samples_per_second_1000'] for r in benchmark_results]
    s10000 = [r['samples_per_second_10000'] for r in benchmark_results]
    
    x = np.arange(len(n_xor_values))
    width = 0.25
    
    plt.bar(x - width, s100, width, label='Batch Size 100')
    plt.bar(x, s1000, width, label='Batch Size 1000')
    plt.bar(x + width, s10000, width, label='Batch Size 10000')
    
    plt.xlabel('XOR PUF Complexity (n)')
    plt.ylabel('Samples per Second')
    plt.title('Generation Speed')
    plt.xticks(x, n_xor_values)
    plt.legend()
    
    # Plot time for largest batch size
    plt.subplot(1, 2, 2)
    times = [r['batch_size_10000'] for r in benchmark_results]
    plt.bar(n_xor_values, times)
    plt.title('Generation Time (10,000 samples)')
    plt.xlabel('XOR PUF Complexity (n)')
    plt.ylabel('Time (seconds)')
    plt.xticks(n_xor_values)
    
    plt.tight_layout()
    plt.savefig('xpuf_gan_generation_benchmark.png')
    plt.close()
    
    return benchmark_results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='XPUF GAN Model')
    parser.add_argument('--data_dir', type=str, default='.', help='Directory containing XOR PUF data files')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'train_all', 'generate', 'benchmark', 'compare'],
                        help='Operation mode')
    parser.add_argument('--file', type=str, help='Specific data file to use (for train mode)')
    parser.add_argument('--n_xor', type=int, help='Number of XOR gates (for generate mode)')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--output_format', type=str, default='hex', choices=['binary', 'hex'],
                       help='Output format for challenges')
    
    args = parser.parse_args()
    
    # Print GPU information
    print("\nGPU Configuration:")
    if len(physical_devices) > 0:
        for i, device in enumerate(physical_devices):
            print(f"  GPU {i}: {device}")
    else:
        print("  No GPUs detected, using CPU")
        
    print("\nTensorFlow version:", tf.__version__)
    print(f"Operation mode: {args.mode}")
    
    if args.mode == 'train':
        if args.file is None:
            print("Error: --file argument is required for train mode")
            return
        
        data_file = os.path.join(args.data_dir, args.file)
        if not os.path.exists(data_file):
            print(f"Error: File {data_file} not found")
            return
        
        # Extract n value from filename
        n = extract_n_from_filename(data_file)
        if n is None:
            print("Could not detect n value from filename, please specify with --n_xor")
            return
        
        print(f"Training {n}-XOR PUF model using {data_file}")
        
        # Load the dataset
        challenges, responses = load_xpuf_dataset(data_file)
        
        # Get challenge dimension from the data
        challenge_dim = challenges.shape[1]
        response_dim = responses.shape[1]
        
        # Initialize and train the GAN
        xpuf_gan = XPUF_GAN(challenge_dim=challenge_dim, response_dim=response_dim, n_xor=n)
        
        # Train the GAN
        xpuf_gan.train(challenges, responses, epochs=args.epochs, batch_size=args.batch_size)
        
        # Generate synthetic dataset
        xpuf_gan.generate_synthetic_dataset(args.samples, output_format=args.output_format)
        
        # Test ML attack resistance
        xpuf_gan.test_ml_attack_resistance(challenges, responses)
        
    elif args.mode == 'train_all':
        print(f"Training models for all XOR PUF configurations in {args.data_dir}")
        train_all_models(args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
        
    elif args.mode == 'generate':
        if args.n_xor is None:
            print("Error: --n_xor argument is required for generate mode")
            return
        
        print(f"Generating {args.samples} synthetic {args.n_xor}-XOR PUF challenge-response pairs")
        
        # Look for the model checkpoint
        model_dir = f"model_checkpoints/{args.n_xor}XOR"
        if not os.path.exists(model_dir):
            print(f"Error: Model directory {model_dir} not found")
            return
        
        # Find the latest model
        generator_files = glob.glob(f"{model_dir}/generator_epoch_*.h5")
        if not generator_files:
            print(f"Error: No generator model found in {model_dir}")
            return
        
        # Get the latest checkpoint
        latest_checkpoint = max(generator_files, key=lambda x: 
                               'final' in x if 'final' in x else 
                               int(re.search(r'epoch_(\d+)', x).group(1)))
        
        epoch = 'final' if 'final' in latest_checkpoint else re.search(r'epoch_(\d+)', latest_checkpoint).group(1)
        print(f"Using checkpoint from epoch {epoch}")
        
        # Look for a data file to get challenge dimension
        data_files = glob.glob(f"{args.data_dir}/{args.n_xor}XOR_*_LUT_*.txt")
        if not data_files:
            print(f"Warning: No dataset found for {args.n_xor}-XOR PUF")
            challenge_dim = 64  # Default value
        else:
            # Load a small sample to get dimensions
            sample_challenges, _ = load_xpuf_dataset(data_files[0])
            challenge_dim = sample_challenges.shape[1]
        
        # Initialize the model
        xpuf_gan = XPUF_GAN(challenge_dim=challenge_dim, response_dim=1, n_xor=args.n_xor)
        
        # Load the trained model
        xpuf_gan.load_model(epoch, checkpoint_dir=model_dir)
        
        # Generate synthetic dataset
        xpuf_gan.generate_synthetic_dataset(args.samples, output_format=args.output_format)
        
    elif args.mode == 'benchmark':
        print("Benchmarking generation performance across all trained models")
        benchmark_generation_performance(args.data_dir)
        
    elif args.mode == 'compare':
        print("Comparing synthetic data across all trained models")
        compare_configurations(args.data_dir, num_samples=args.samples)
        
    print("\nOperation completed successfully")


if __name__ == "__main__":
    main()