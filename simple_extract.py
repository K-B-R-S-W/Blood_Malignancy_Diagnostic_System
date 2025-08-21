#!/usr/bin/env python3
"""
Simple script to extract basic results from blood_cell_results.pth
Requires only matplotlib (which comes with most Python installations)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def main():
    print("🔬 Loading Blood Cell Training Results...")
    
    # Load the results
    try:
        results = torch.load('model/blood_cell_results.pth', map_location='cpu')
        print("✅ Results loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_png_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 Available data in results file:")
    for key, value in results.items():
        print(f"   {key}: {type(value)}")
    
    # Plot training curves if available
    if 'train_losses' in results and 'val_losses' in results:
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        epochs = range(1, len(results['train_losses']) + 1)
        plt.plot(epochs, results['train_losses'], 'b-', label='Training Loss')
        plt.plot(epochs, results['val_losses'], 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        if 'train_accs' in results and 'val_accs' in results:
            plt.plot(epochs, [acc * 100 for acc in results['train_accs']], 'b-', label='Training Accuracy')
            plt.plot(epochs, [acc * 100 for acc in results['val_accs']], 'r-', label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Training curves saved")
    
    # Show final metrics
    if 'test_accuracy' in results:
        print(f"\n🎯 Final Test Accuracy: {results['test_accuracy']*100:.2f}%")
    
    if 'class_names' in results:
        print(f"📝 Classes: {', '.join(results['class_names'])}")
    
    if 'class_accuracies' in results:
        print(f"\n📊 Per-class accuracies:")
        for class_name, accuracy in results['class_accuracies'].items():
            print(f"   {class_name.capitalize()}: {accuracy*100:.2f}%")
    
    print(f"\n✅ Results extracted to: {output_dir}/")

if __name__ == "__main__":
    main()
