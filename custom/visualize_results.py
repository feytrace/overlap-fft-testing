#!/usr/bin/env python3
"""
Visualization tool for CUDA 2D Poisson Solver Results
Generates comparison plots between computed and analytical solutions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys

def generate_analytical_solution(Nx, Ny):
    """Generate the analytical solution for comparison"""
    hx = 1.0 / (Nx + 1)
    hy = 1.0 / (Ny + 1)
    
    x = np.linspace(hx, 1.0 - hx, Nx)
    y = np.linspace(hy, 1.0 - hy, Ny)
    X, Y = np.meshgrid(x, y)
    
    # Analytical solution: u(x,y) = sin(pi*x)*sin(pi*y)
    U_analytical = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    return X, Y, U_analytical

def read_solution_from_file(filename, Nx, Ny):
    """Read solution from binary or text file"""
    try:
        # Try reading as binary double precision
        data = np.fromfile(filename, dtype=np.float64)
        if len(data) == Nx * Ny:
            return data.reshape((Ny, Nx))
    except:
        pass
    
    try:
        # Try reading as text
        data = np.loadtxt(filename)
        if data.size == Nx * Ny:
            return data.reshape((Ny, Nx))
    except:
        pass
    
    return None

def create_comparison_plots(Nx, Ny, U_computed=None, save_prefix="solver_results"):
    """Create comprehensive comparison plots"""
    
    # Generate analytical solution
    X, Y, U_analytical = generate_analytical_solution(Nx, Ny)
    
    # If no computed solution provided, use analytical for demonstration
    if U_computed is None:
        print("No computed solution provided, using analytical solution for demonstration")
        U_computed = U_analytical
    
    # Compute error
    error = U_computed - U_analytical
    error_abs = np.abs(error)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # ============================================================
    # 1. 3D Surface Plot - Analytical Solution
    # ============================================================
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, U_analytical, cmap=cm.viridis, 
                             linewidth=0, antialiased=True, alpha=0.9)
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.set_zlabel('U', fontsize=10)
    ax1.set_title('Analytical Solution\n$u(x,y) = sin(\pi x)sin(\pi y)$', fontsize=11, fontweight='bold')
    ax1.view_init(elev=30, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # ============================================================
    # 2. 3D Surface Plot - Computed Solution
    # ============================================================
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, U_computed, cmap=cm.viridis, 
                             linewidth=0, antialiased=True, alpha=0.9)
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Y', fontsize=10)
    ax2.set_zlabel('U', fontsize=10)
    ax2.set_title('Computed Solution\n(CUDA Solver)', fontsize=11, fontweight='bold')
    ax2.view_init(elev=30, azim=45)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # ============================================================
    # 3. 3D Surface Plot - Absolute Error
    # ============================================================
    ax3 = fig.add_subplot(2, 4, 3, projection='3d')
    surf3 = ax3.plot_surface(X, Y, error_abs, cmap=cm.hot, 
                             linewidth=0, antialiased=True, alpha=0.9)
    ax3.set_xlabel('X', fontsize=10)
    ax3.set_ylabel('Y', fontsize=10)
    ax3.set_zlabel('Error', fontsize=10)
    ax3.set_title('Absolute Error\nMax: {:.2e}'.format(np.max(error_abs)), fontsize=11, fontweight='bold')
    ax3.view_init(elev=30, azim=45)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    # ============================================================
    # 4. Contour Plot - Analytical Solution
    # ============================================================
    ax4 = fig.add_subplot(2, 4, 4)
    contour1 = ax4.contourf(X, Y, U_analytical, levels=20, cmap=cm.viridis)
    ax4.set_xlabel('X', fontsize=10)
    ax4.set_ylabel('Y', fontsize=10)
    ax4.set_title('Analytical (Contour)', fontsize=11, fontweight='bold')
    ax4.set_aspect('equal')
    fig.colorbar(contour1, ax=ax4)
    
    # ============================================================
    # 5. Contour Plot - Computed Solution
    # ============================================================
    ax5 = fig.add_subplot(2, 4, 5)
    contour2 = ax5.contourf(X, Y, U_computed, levels=20, cmap=cm.viridis)
    ax5.set_xlabel('X', fontsize=10)
    ax5.set_ylabel('Y', fontsize=10)
    ax5.set_title('Computed (Contour)', fontsize=11, fontweight='bold')
    ax5.set_aspect('equal')
    fig.colorbar(contour2, ax=ax5)
    
    # ============================================================
    # 6. Contour Plot - Error
    # ============================================================
    ax6 = fig.add_subplot(2, 4, 6)
    contour3 = ax6.contourf(X, Y, error_abs, levels=20, cmap=cm.hot)
    ax6.set_xlabel('X', fontsize=10)
    ax6.set_ylabel('Y', fontsize=10)
    ax6.set_title('Absolute Error (Contour)', fontsize=11, fontweight='bold')
    ax6.set_aspect('equal')
    fig.colorbar(contour3, ax=ax6)
    
    # ============================================================
    # 7. Cross-section at y=0.5
    # ============================================================
    ax7 = fig.add_subplot(2, 4, 7)
    mid_y = Ny // 2
    ax7.plot(X[mid_y, :], U_analytical[mid_y, :], 'b-', linewidth=2, label='Analytical')
    ax7.plot(X[mid_y, :], U_computed[mid_y, :], 'r--', linewidth=2, label='Computed')
    ax7.set_xlabel('X', fontsize=10)
    ax7.set_ylabel('U', fontsize=10)
    ax7.set_title('Cross-section at y={:.2f}'.format(Y[mid_y, 0]), fontsize=11, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # ============================================================
    # 8. Error Statistics
    # ============================================================
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    
    # Compute statistics
    max_error = np.max(error_abs)
    mean_error = np.mean(error_abs)
    l2_error = np.sqrt(np.mean(error**2))
    relative_error = l2_error / np.sqrt(np.mean(U_analytical**2))
    
    stats_text = """
    ERROR STATISTICS
    ════════════════════════════════
    
    Grid Size: {} × {}
    Total Points: {:,}
    
    Max Absolute Error:
        {:.6e}
    
    Mean Absolute Error:
        {:.6e}
    
    L2 Error Norm:
        {:.6e}
    
    Relative L2 Error:
        {:.6e}
        ({:.4f}%)
    
    Min Computed: {:.6f}
    Max Computed: {:.6f}
    
    Min Analytical: {:.6f}
    Max Analytical: {:.6f}
    """.format(Nx, Ny, Nx * Ny, max_error, mean_error, l2_error, 
               relative_error, relative_error*100, np.min(U_computed), 
               np.max(U_computed), np.min(U_analytical), np.max(U_analytical))
    
    ax8.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.3))
    
    plt.suptitle('CUDA 2D Poisson Solver - Results Comparison', 
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

def create_convergence_plot(sizes, errors, save_name="convergence.png"):
    """Create convergence plot for different grid sizes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Log-log plot
    ax1.loglog(sizes, errors, 'bo-', linewidth=2, markersize=8, label='Computed Error')
    
    # Add reference lines
    h = 1.0 / np.array(sizes)
    ax1.loglog(sizes, h**2 * errors[0] / (h[0]**2), 'r--', 
               linewidth=1.5, label='$O(h^2)$ reference')
    
    ax1.set_xlabel('Grid Size N', fontsize=12)
    ax1.set_ylabel('L2 Error', fontsize=12)
    ax1.set_title('Convergence Analysis (Log-Log)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Convergence rate plot
    if len(sizes) > 1:
        rates = []
        for i in range(1, len(sizes)):
            rate = np.log(errors[i-1] / errors[i]) / np.log(sizes[i] / sizes[i-1])
            rates.append(rate)
        
        ax2.plot(sizes[1:], rates, 'go-', linewidth=2, markersize=8)
        ax2.axhline(y=2.0, color='r', linestyle='--', linewidth=1.5, label='Expected: 2.0')
        ax2.set_xlabel('Grid Size N', fontsize=12)
        ax2.set_ylabel('Convergence Rate', fontsize=12)
        ax2.set_title('Observed Convergence Rate', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print("✓ Saved convergence plot to: {}".format(save_name))
    
    return fig

def main():
    """Main function"""
    print("=" * 60)
    print("CUDA 2D Poisson Solver - Visualization Tool")
    print("=" * 60)
    
    # Default parameters
    Nx = 100
    Ny = 100
    solution_file = None
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        Nx = int(sys.argv[1])
    if len(sys.argv) > 2:
        Ny = int(sys.argv[2])
    if len(sys.argv) > 3:
        solution_file = sys.argv[3]
    
    print("\nGrid Size: {} × {}".format(Nx, Ny))
    
    # Read computed solution if provided
    U_computed = None
    if solution_file:
        print("Reading solution from: {}".format(solution_file))
        U_computed = read_solution_from_file(solution_file, Nx, Ny)
        if U_computed is None:
            print("Warning: Could not read solution file, using analytical solution")
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    fig = create_comparison_plots(Nx, Ny, U_computed)
    
    # Show plot
    print("\nDisplaying plots...")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
