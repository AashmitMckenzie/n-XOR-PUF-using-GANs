import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import json
from datetime import datetime
import pandas as pd
import re

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the base output directory
OUTPUT_DIR = r"C:\Aashmit.2\Important\DEPRESSION\CODING\Cyber Security\GANs PUF"

class XPUF_GAN:
    def __init__(self, challenge_dim=64, response_dim=1, latent_dim=100, output_dir=OUTPUT_DIR):
        """
        Initialize the XPUF GAN model
        
        Args:
            challenge_dim: Dimension of the PUF challenge vector
            response_dim: Dimension of the PUF response vector
            latent_dim: Dimension of the random noise input
            output_dir: Directory to save all outputs
        """
        self.challenge_dim = challenge_dim
        self.response_dim = response_dim
        self.latent_dim = latent_dim
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
                                  optimizer=Adam(0.0002, 0.5),
                                  metrics=['accuracy'])
        
        # Build the generator
        self.generator = self.build_generator()
        
        # Build the combined model
        # Freeze discriminator weights for GAN
        self.combined = self.build_gan()
        self.combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

        # Re-enable training on discriminator (for future training calls)
        self.discriminator.trainable = True
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=Adam(0.0002, 0.5),
            metrics=['accuracy']
        )

        
    def build_generator(self):
        """Build the Generator model for XPUF CRP synthesis"""
        # Define model inputs
        noise_input = Input(shape=(self.latent_dim,))
        challenge_input = Input(shape=(self.challenge_dim,))
        
        # Merge inputs
        merged = Concatenate()([noise_input, challenge_input])
        
        # Dense layers with batch normalization and leaky ReLU
        x = Dense(256)(merged)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Dense(512)(merged)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        # Output layer for response generation (tanh activation for -1 to 1 range)
        response_output = Dense(self.response_dim, activation='tanh')(x)
        
        # Define generator model
        model = Model([noise_input, challenge_input], response_output)
        return model
    
    def build_discriminator(self):
        """Build a Discriminator model to distinguish real/fake XPUF CRPs"""
        # Inputs
        challenge_input = Input(shape=(self.challenge_dim,))
        response_input = Input(shape=(self.response_dim,))
        
        # Merge inputs
        merged = Concatenate()([challenge_input, response_input])
        
        # Dense layers with dropout and leaky ReLU
        x = Dense(512)(merged)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        
        x = Dense(256)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        
        x = Dense(128)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        
        # Output layer - real or fake
        validity = Dense(1, activation='sigmoid')(x)
        
        # Define discriminator model
        model = Model([challenge_input, response_input], validity)
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
        model = Model([noise, challenge], validity)
        return model
    
    def train(self, real_challenges, real_responses, epochs=10000, batch_size=128, 
              save_interval=1000, validation_interval=500):
        """
        Train the GAN model
        
        Args:
            real_challenges: Array of real PUF challenges
            real_responses: Array of real PUF responses
            epochs: Number of training epochs
            batch_size: Training batch size
            save_interval: Interval at which to save model checkpoints
            validation_interval: Interval at which to validate the model
        """
        # Create directory for checkpoints if it doesn't exist
        checkpoint_dir = os.path.join(self.output_dir, 'model_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Normalize responses to [-1, 1] for tanh activation
        real_responses_normalized = real_responses * 2 - 1
        
        # Training history
        d_losses = []
        g_losses = []
        d_accuracies = []
        
        # Create directory for results if it doesn't exist
        results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Define adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select a random batch of real CRPs
            idx = np.random.randint(0, real_challenges.shape[0], batch_size)
            batch_challenges = real_challenges[idx]
            batch_responses = real_responses_normalized[idx]
            
            # Sample noise and generate a batch of new responses
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_responses = self.generator.predict([noise, batch_challenges])
            
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
            
            # Safely convert g_loss to a scalar
            try:
                g_loss_value = float(g_loss[0])  # if it's list/array
            except (TypeError, IndexError):
                g_loss_value = float(g_loss)     # if it's a scalar

            g_losses.append(g_loss_value)
            
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, [D loss: {d_loss[0]:.4f}, acc.: {d_loss[1]*100:.2f}%] [G loss: {g_loss_value:.4f}]")

            
            # Periodically save model checkpoints
            if epoch % save_interval == 0:
                self.save_model(epoch)
                
                # Plot training progress
                self.plot_training_progress(d_losses, g_losses, d_accuracies, epoch)
            
            # Validate model periodically
            if epoch % validation_interval == 0 and epoch > 0:
                self.validate_model(real_challenges, real_responses_normalized, epoch)
                
        return d_losses, g_losses, d_accuracies
    
    def save_model(self, epoch):
        """Save model checkpoints"""
        checkpoint_dir = os.path.join(self.output_dir, 'model_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.generator.save(os.path.join(checkpoint_dir, f'generator_epoch_{epoch}.h5'))
        self.discriminator.save(os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch}.h5'))
    
    def plot_training_progress(self, d_losses, g_losses, d_accuracies, epoch):
        """Plot and save training progress"""
        results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 5))
        
        # Plot discriminator and generator loss
        plt.subplot(1, 2, 1)
        plt.plot(d_losses, label='Discriminator')
        plt.plot(g_losses, label='Generator')
        plt.title('Model Losses')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot discriminator accuracy
        plt.subplot(1, 2, 2)
        plt.plot(d_accuracies)
        plt.title('Discriminator Accuracy')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Epoch')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'training_progress_epoch_{epoch}.png'))
        plt.close()
    
    def validate_model(self, real_challenges, real_responses, epoch):
        """Validate the model by comparing synthetic and real responses"""
        results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate synthetic responses for validation
        num_validation = min(1000, len(real_challenges))
        validation_challenges = real_challenges[:num_validation]
        noise = np.random.normal(0, 1, (num_validation, self.latent_dim))
        
        synthetic_responses = self.generator.predict([noise, validation_challenges])
        # Convert back from [-1, 1] to [0, 1]
        synthetic_responses_binary = (synthetic_responses > 0).astype(int)
        real_responses_binary = (real_responses > 0).astype(int)
        
        # Calculate validation metrics
        validation_metrics = self.calculate_validation_metrics(
            real_responses_binary[:num_validation], synthetic_responses_binary)
        
        # Save validation metrics
        with open(os.path.join(results_dir, f'validation_metrics_epoch_{epoch}.json'), 'w') as f:
            json.dump(validation_metrics, f)
            
        print(f"Epoch {epoch} - Validation Metrics:")
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
    
    def generate_synthetic_dataset(self, num_samples, output_format='binary'):
        """
        Generate a complete synthetic XPUF dataset
        
        Args:
            num_samples: Number of synthetic samples to generate
            output_format: Format of challenges in output ('binary' or 'hex')
            
        Returns:
            Tuple of (challenges, responses, metadata)
        """
        # Generate random challenges
        synthetic_challenges = np.random.randint(0, 2, (num_samples, self.challenge_dim))
        
        # Generate random latent space vectors
        latent_vectors = np.random.normal(0, 1, (num_samples, self.latent_dim))
        
        # Generate responses
        synthetic_responses = self.generator.predict([latent_vectors, synthetic_challenges])
        
        # Convert from [-1, 1] to binary [0, 1]
        synthetic_responses_binary = (synthetic_responses > 0).astype(int)
        
        # Create metadata
        metadata = {
            'date_generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_samples': num_samples,
            'challenge_dim': self.challenge_dim,
            'response_dim': self.response_dim,
            'validation_metrics': self.calculate_validation_metrics(
                synthetic_responses_binary[:100], synthetic_responses_binary[-100:])
        }
        
        # Save dataset
        synthetic_dir = os.path.join(self.output_dir, 'synthetic_dataset')
        os.makedirs(synthetic_dir, exist_ok=True)
        
        # Save in binary format
        np.save(os.path.join(synthetic_dir, 'synthetic_xpuf_challenges.npy'), synthetic_challenges)
        np.save(os.path.join(synthetic_dir, 'synthetic_xpuf_responses.npy'), synthetic_responses_binary)
        
        # Also save in the same format as the original dataset
        with open(os.path.join(synthetic_dir, 'synthetic_xpuf_crps.txt'), 'w') as f:
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
        
        with open(os.path.join(synthetic_dir, 'synthetic_xpuf_metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        print(f"Generated {num_samples} synthetic XPUF challenge-response pairs")
        print(f"Dataset saved in '{synthetic_dir}' directory")
        
        return synthetic_challenges, synthetic_responses_binary, metadata
    
    def test_ml_attack_resistance(self, real_challenges, real_responses):
        """Test ML attack resistance by training a model to predict responses"""
        results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate synthetic dataset
        num_samples = real_challenges.shape[0]
        synthetic_challenges, synthetic_responses, _ = self.generate_synthetic_dataset(num_samples)
        
        # Build simple ML attack model
        def build_ml_attack_model(input_dim, output_dim):
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
        model_real = build_ml_attack_model(X_train_real.shape[1], y_train_real.shape[1])
        model_syn = build_ml_attack_model(X_train_syn.shape[1], y_train_syn.shape[1])
        
        # Train on real data
        model_real.fit(X_train_real, y_train_real, epochs=50, batch_size=64, verbose=0)
        
        # Train on synthetic data
        model_syn.fit(X_train_syn, y_train_syn, epochs=50, batch_size=64, verbose=0)
        
        # Evaluate both models on real test data
        real_on_real = model_real.evaluate(X_test_real, y_test_real)[1]  # accuracy
        syn_on_real = model_syn.evaluate(X_test_real, y_test_real)[1]  # accuracy
        
        # Evaluate on synthetic test data
        real_on_syn = model_real.evaluate(X_test_syn, y_test_syn)[1]  # accuracy
        syn_on_syn = model_syn.evaluate(X_test_syn, y_test_syn)[1]  # accuracy
        
        results = {
            'real_model_on_real_data': float(real_on_real),
            'synthetic_model_on_real_data': float(syn_on_real),
            'real_model_on_synthetic_data': float(real_on_syn),
            'synthetic_model_on_synthetic_data': float(syn_on_syn),
            'difference_on_real_data': float(abs(real_on_real - syn_on_real))
        }
        
        # Save results to file
        with open(os.path.join(results_dir, 'ml_attack_results.json'), 'w') as f:
            json.dump(results, f)
        
        print("ML Attack Success Rate:")
        print(f"  Model trained on real data, tested on real data: {real_on_real:.4f}")
        print(f"  Model trained on synthetic data, tested on real data: {syn_on_real:.4f}")
        print(f"  Model trained on real data, tested on synthetic data: {real_on_syn:.4f}")
        print(f"  Model trained on synthetic data, tested on synthetic data: {syn_on_syn:.4f}")
        print(f"  Difference on real data: {abs(real_on_real - syn_on_real):.4f}")
        
        return results

    def reliability_analysis(self, challenges, noise_levels=[0.01, 0.05, 0.1]):
        """Analyze reliability of synthetic PUF responses under noise"""
        results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        num_samples = min(1000, challenges.shape[0])
        challenges = challenges[:num_samples]
        
        # Generate baseline responses
        baseline_latent = np.random.normal(0, 1, (num_samples, self.latent_dim))
        baseline_responses = self.generator.predict([baseline_latent, challenges])
        baseline_binary = (baseline_responses > 0).astype(int)
        
        reliability_scores = []
        reliability_results = {}
        
        for noise_level in noise_levels:
            bit_errors = []
            
            # Generate multiple noisy responses for each challenge
            for _ in range(10):
                # Add noise to the latent space
                noisy_latent = baseline_latent.copy()
                noisy_latent += np.random.normal(0, noise_level, noisy_latent.shape)
                
                # Generate responses with noisy latent
                noisy_resp = self.generator.predict([noisy_latent, challenges])
                noisy_binary = (noisy_resp > 0).astype(int)
                
                # Calculate bit error rate
                bit_error = np.mean(np.abs(baseline_binary - noisy_binary))
                bit_errors.append(bit_error)
            
            avg_bit_error = np.mean(bit_errors)
            reliability = 1 - avg_bit_error
            reliability_scores.append(reliability)
            reliability_results[f"noise_level_{noise_level}"] = float(reliability)
            
            print(f"Noise level {noise_level}: Reliability = {reliability*100:.2f}%")
        
        # Save reliability results
        with open(os.path.join(results_dir, 'reliability_results.json'), 'w') as f:
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
                hex_challenge = parts[0]
                response = int(parts[1])
                
                # Convert hex to binary
                bin_challenge = bin(int(hex_challenge, 16))[2:]
                
                # Pad to ensure correct length
                challenge_dim = len(hex_challenge) * 4  # Each hex digit is 4 bits
                bin_challenge = bin_challenge.zfill(challenge_dim)
                
                # Reverse the order as specified
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


def main():
    # Define output directory
    output_dir = r"C:\Aashmit.2\Important\DEPRESSION\CODING\Cyber Security\GANs PUF"
    os.makedirs(output_dir, exist_ok=True)
    
    # Example usage
    data_file = r"C:\Aashmit.2\Important\DEPRESSION\CODING\Cyber Security\GANs PUF\4XOR_64bit_LUT_2239B_attacking_1M.txt"  # Replace with your actual file path
    
    # Extract n value from filename
    n = extract_n_from_filename(data_file)
    if n:
        print(f"Detected {n}-XOR PUF from filename")
    else:
        print("Could not detect n value, assuming n=4")
        n = 4
    
    # Load the dataset
    challenges, responses = load_xpuf_dataset(data_file)
    challenges = challenges.astype(np.float32)
    responses = responses.astype(np.float32)
    # Get challenge dimension from the data
    challenge_dim = challenges.shape[1]
    response_dim = responses.shape[1]
    

    # Initialize and train the GAN
    xpuf_gan = XPUF_GAN(challenge_dim=challenge_dim, response_dim=response_dim, output_dir=output_dir)
    
    # Train the GAN
    xpuf_gan.train(challenges, responses, epochs=500, batch_size=128, 
                   save_interval=1000, validation_interval=500)
    
    # Generate synthetic dataset - generate same number of samples as original
    num_samples = len(challenges)
    synthetic_challenges, synthetic_responses, metadata = xpuf_gan.generate_synthetic_dataset(
        num_samples, output_format='hex')
    
    # Test ML attack resistance
    ml_results = xpuf_gan.test_ml_attack_resistance(challenges, responses)
    
    # Reliability analysis
    reliability_scores = xpuf_gan.reliability_analysis(challenges[:1000])
    
    # Save model architecture diagram
    model_summary = {
        'generator': [],
        'discriminator': []
    }
    
    # Capture generator summary
    string_buffer = []
    xpuf_gan.generator.summary(print_fn=lambda x: string_buffer.append(x))
    model_summary['generator'] = '\n'.join(string_buffer)
    
    # Capture discriminator summary
    string_buffer = []
    xpuf_gan.discriminator.summary(print_fn=lambda x: string_buffer.append(x))
    model_summary['discriminator'] = '\n'.join(string_buffer)
    
    # Save model summaries
    with open(os.path.join(output_dir, 'model_summary.txt'), 'w') as f:
        f.write(f"{n}-XOR PUF GAN Architecture\n")
        f.write("="*50 + "\n\n")
        f.write("Generator Architecture:\n")
        f.write("-"*50 + "\n")
        f.write(model_summary['generator'])
        f.write("\n\n" + "="*50 + "\n\n")
        f.write("Discriminator Architecture:\n")
        f.write("-"*50 + "\n")
        f.write(model_summary['discriminator'])


if __name__ == "__main__":
    main()