#!/usr/bin/env python3
"""
Visualization tool for CUDA Gaussian Process Results
Generates comparison plots between GP predictions and true function
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys

def generate_true_function(X, Y):
    """Generate the true function: f(x,y) = sin(x) * sin(2y)"""
    return np.sin(X) * np.sin(2.0 * Y)

def read_data_from_file(filename):
    """Read GP training/test data from file"""
    try:
        data = np.loadtxt(filename)
        return data
    except:
        print("Error reading file: {}".format(filename))
        return None

def create_gp_visualization(X_train=None, Y_train=None, f_train=None,
                            X_test=None, Y_test=None, mu_pred=None, 
                            save_prefix="gp_results"):
    """Create comprehensive GP visualization"""
    
    # If no data provided, generate synthetic data for demonstration
    if X_train is None:
        print("Generating synthetic data for demonstration...")
        np.random.seed(42)
        N_train = 50
        X_train = np.random.rand(N_train)
        Y_train = np.random.rand(N_train)
        f_train = generate_true_function(X_train, Y_train)
        f_train += np.random.randn(N_train) * 0.1  # Add noise
        
        N_test = 30
        X_test = np.linspace(0, 1, N_test)
        Y_test = np.ones(N_test) * 0.5
        mu_pred = generate_true_function(X_test, Y_test)
    
    # Create fine grid for true function surface
    x_grid = np.linspace(0, 1, 100)
    y_grid = np.linspace(0, 1, 100)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    Z_true = generate_true_function(X_grid, Y_grid)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # ============================================================
    # 1. 3D Surface - True Function
    # ============================================================
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    surf1 = ax1.plot_surface(X_grid, Y_grid, Z_true, cmap=cm.viridis,
                             linewidth=0, antialiased=True, alpha=0.8)
    ax1.scatter(X_train, Y_train, f_train, c='red', s=50, 
                marker='o', edgecolors='black', linewidths=1.5,
                label='Training Data')
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.set_zlabel('f(x,y)', fontsize=10)
    ax1.set_title('True Function: sin(x) * sin(2y)\n+ Training Points', 
                  fontsize=11, fontweight='bold')
    ax1.view_init(elev=25, azim=45)
    ax1.legend(loc='upper right', fontsize=8)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # ============================================================
    # 2. 3D Surface - True Function (Different Angle)
    # ============================================================
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    surf2 = ax2.plot_surface(X_grid, Y_grid, Z_true, cmap=cm.viridis,
                             linewidth=0, antialiased=True, alpha=0.8)
    ax2.scatter(X_train, Y_train, f_train, c='red', s=50, 
                marker='o', edgecolors='black', linewidths=1.5)
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Y', fontsize=10)
    ax2.set_zlabel('f(x,y)', fontsize=10)
    ax2.set_title('True Function (Rotated View)', fontsize=11, fontweight='bold')
    ax2.view_init(elev=25, azim=135)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # ============================================================
    # 3. Contour Plot - True Function with Training Points
    # ============================================================
    ax3 = fig.add_subplot(2, 4, 3)
    contour1 = ax3.contourf(X_grid, Y_grid, Z_true, levels=20, cmap=cm.viridis)
    ax3.scatter(X_train, Y_train, c=f_train, s=100, 
                marker='o', edgecolors='black', linewidths=2,
                cmap=cm.viridis, vmin=Z_true.min(), vmax=Z_true.max())
    ax3.set_xlabel('X', fontsize=10)
    ax3.set_ylabel('Y', fontsize=10)
    ax3.set_title('True Function (Contour)\n+ Training Data', 
                  fontsize=11, fontweight='bold')
    ax3.set_aspect('equal')
    fig.colorbar(contour1, ax=ax3)
    
    # ============================================================
    # 4. GP Prediction along y=0.5 slice
    # ============================================================
    ax4 = fig.add_subplot(2, 4, 4)
    # True function along slice
    x_slice = np.linspace(0, 1, 200)
    y_slice = 0.5
    z_slice_true = generate_true_function(x_slice, y_slice)
    
    ax4.plot(x_slice, z_slice_true, 'b-', linewidth=2.5, label='True Function')
    
    # Training points on this slice (within tolerance)
    tolerance = 0.1
    mask = np.abs(Y_train - y_slice) < tolerance
    if np.any(mask):
        ax4.scatter(X_train[mask], f_train[mask], c='red', s=100, 
                   marker='o', edgecolors='black', linewidths=2,
                   label='Training Data', zorder=5)
    
    # GP predictions
    if X_test is not None and mu_pred is not None:
        mask_test = np.abs(Y_test - y_slice) < tolerance
        if np.any(mask_test):
            ax4.plot(X_test[mask_test], mu_pred[mask_test], 'g--', 
                    linewidth=2, label='GP Prediction', alpha=0.8)
            ax4.scatter(X_test[mask_test], mu_pred[mask_test], c='green', 
                       s=60, marker='s', edgecolors='black', linewidths=1.5,
                       zorder=5)
    
    ax4.set_xlabel('X', fontsize=11)
    ax4.set_ylabel('f(x, 0.5)', fontsize=11)
    ax4.set_title('Cross-section at y = 0.5', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    
    # ============================================================
    # 5. GP Prediction along x=0.5 slice
    # ============================================================
    ax5 = fig.add_subplot(2, 4, 5)
    # True function along slice
    x_slice2 = 0.5
    y_slice2 = np.linspace(0, 1, 200)
    z_slice_true2 = generate_true_function(x_slice2, y_slice2)
    
    ax5.plot(y_slice2, z_slice_true2, 'b-', linewidth=2.5, label='True Function')
    
    # Training points on this slice
    mask2 = np.abs(X_train - x_slice2) < tolerance
    if np.any(mask2):
        ax5.scatter(Y_train[mask2], f_train[mask2], c='red', s=100, 
                   marker='o', edgecolors='black', linewidths=2,
                   label='Training Data', zorder=5)
    
    # GP predictions
    if X_test is not None and mu_pred is not None:
        mask_test2 = np.abs(X_test - x_slice2) < tolerance
        if np.any(mask_test2):
            ax5.plot(Y_test[mask_test2], mu_pred[mask_test2], 'g--', 
                    linewidth=2, label='GP Prediction', alpha=0.8)
            ax5.scatter(Y_test[mask_test2], mu_pred[mask_test2], c='green', 
                       s=60, marker='s', edgecolors='black', linewidths=1.5,
                       zorder=5)
    
    ax5.set_xlabel('Y', fontsize=11)
    ax5.set_ylabel('f(0.5, y)', fontsize=11)
    ax5.set_title('Cross-section at x = 0.5', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9, loc='best')
    ax5.grid(True, alpha=0.3)
    
    # ============================================================
    # 6. Training Data Distribution
    # ============================================================
    ax6 = fig.add_subplot(2, 4, 6)
    scatter = ax6.scatter(X_train, Y_train, c=f_train, s=150, 
                          marker='o', edgecolors='black', linewidths=2,
                          cmap=cm.viridis)
    ax6.set_xlabel('X', fontsize=11)
    ax6.set_ylabel('Y', fontsize=11)
    ax6.set_title('Training Data Distribution\n({} points)'.format(len(X_train)), 
                  fontsize=11, fontweight='bold')
    ax6.set_xlim([0, 1])
    ax6.set_ylim([0, 1])
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=ax6, label='f(x,y)')
    
    # ============================================================
    # 7. Prediction Error (if test data available)
    # ============================================================
    ax7 = fig.add_subplot(2, 4, 7)
    if X_test is not None and mu_pred is not None:
        # Compute true values at test points
        f_test_true = generate_true_function(X_test, Y_test)
        errors = np.abs(mu_pred - f_test_true)
        
        scatter7 = ax7.scatter(X_test, Y_test, c=errors, s=150,
                              marker='s', edgecolors='black', linewidths=2,
                              cmap=cm.hot)
        ax7.set_xlabel('X', fontsize=11)
        ax7.set_ylabel('Y', fontsize=11)
        ax7.set_title('Absolute Prediction Error\nMax: {:.4f}'.format(np.max(errors)), 
                      fontsize=11, fontweight='bold')
        ax7.set_xlim([0, 1])
        ax7.set_ylim([0, 1])
        ax7.set_aspect('equal')
        ax7.grid(True, alpha=0.3)
        fig.colorbar(scatter7, ax=ax7, label='|error|')
    else:
        ax7.text(0.5, 0.5, 'No test data available', 
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, transform=ax7.transAxes)
        ax7.axis('off')
    
    # ============================================================
    # 8. Statistics
    # ============================================================
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    
    # Compute statistics
    stats_text = """
    GAUSSIAN PROCESS STATISTICS
    ════════════════════════════════
    
    Training Data:
        Points: {}
        X range: [{:.3f}, {:.3f}]
        Y range: [{:.3f}, {:.3f}]
        f range: [{:.3f}, {:.3f}]
    """.format(len(X_train), 
               np.min(X_train), np.max(X_train),
               np.min(Y_train), np.max(Y_train),
               np.min(f_train), np.max(f_train))
    
    if X_test is not None and mu_pred is not None:
        f_test_true = generate_true_function(X_test, Y_test)
        errors = np.abs(mu_pred - f_test_true)
        rmse = np.sqrt(np.mean((mu_pred - f_test_true)**2))
        mae = np.mean(errors)
        max_error = np.max(errors)
        
        stats_text += """
    Test Data:
        Points: {}
        RMSE: {:.6f}
        MAE: {:.6f}
        Max Error: {:.6f}
    
    True Function:
        f(x,y) = sin(x) * sin(2y)
        
    Kernel:
        Cosine kernel
        K(x,x') = σ² cos(k₀ₓΔx) cos(k₀ᵧΔy)
        """.format(len(X_test), rmse, mae, max_error)
    else:
        stats_text += """
    
    True Function:
        f(x,y) = sin(x) * sin(2y)
        
    Kernel:
        Cosine kernel
        K(x,x') = σ² cos(k₀ₓΔx) cos(k₀ᵧΔy)
        """
    
    ax8.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Gaussian Process with Cosine Kernel - Visualization', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_file = "{}.png".format(save_prefix)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to: {}".format(output_file))
    
    # Also save high-resolution version
    output_file_hires = "{}_hires.png".format(save_prefix)
    plt.savefig(output_file_hires, dpi=300, bbox_inches='tight')
    print("✓ Saved high-res version to: {}".format(output_file_hires))
    
    return fig

def create_convergence_plot(N_trains, rmses, maes, save_name="gp_convergence.png"):
    """Create convergence plot for different training set sizes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # RMSE vs training size
    ax1.plot(N_trains, rmses, 'bo-', linewidth=2, markersize=8, label='RMSE')
    ax1.plot(N_trains, maes, 'rs-', linewidth=2, markersize=8, label='MAE')
    ax1.set_xlabel('Number of Training Points', fontsize=12)
    ax1.set_ylabel('Error', fontsize=12)
    ax1.set_title('GP Error vs Training Size', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Log-log plot
    ax2.loglog(N_trains, rmses, 'bo-', linewidth=2, markersize=8, label='RMSE')
    ax2.loglog(N_trains, maes, 'rs-', linewidth=2, markersize=8, label='MAE')
    ax2.set_xlabel('Number of Training Points', fontsize=12)
    ax2.set_ylabel('Error (log scale)', fontsize=12)
    ax2.set_title('GP Convergence (Log-Log)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print("✓ Saved convergence plot to: {}".format(save_name))
    
    return fig

def main():
    """Main function"""
    print("=" * 60)
    print("CUDA Gaussian Process - Visualization Tool")
    print("=" * 60)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        train_file = sys.argv[1]
        print("\nReading training data from: {}".format(train_file))
        train_data = read_data_from_file(train_file)
        
        if train_data is not None and train_data.shape[1] >= 3:
            X_train = train_data[:, 0]
            Y_train = train_data[:, 1]
            f_train = train_data[:, 2]
            
            X_test = None
            Y_test = None
            mu_pred = None
            
            if len(sys.argv) > 2:
                test_file = sys.argv[2]
                print("Reading test data from: {}".format(test_file))
                test_data = read_data_from_file(test_file)
                
                if test_data is not None and test_data.shape[1] >= 3:
                    X_test = test_data[:, 0]
                    Y_test = test_data[:, 1]
                    mu_pred = test_data[:, 2]
            
            print("\nGenerating visualization...")
            fig = create_gp_visualization(X_train, Y_train, f_train,
                                         X_test, Y_test, mu_pred)
        else:
            print("Error: Invalid data format")
            print("Expected format: X Y f (one point per line)")
            return
    else:
        print("\nNo data files provided, generating demonstration...")
        fig = create_gp_visualization()
    
    # Show plot
    print("\nDisplaying plots...")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
