# Time Complexity Analysis: Arbitrary_Shape_Parameterized

## Variables
- **N** = number of particles
- **M** = number of cells (triangles in mesh)
- **n_tot** = number of time steps
- **R** = number of particles to rebin per time step (typically R << N)
- **P_avg** = average particles per cell = N/M
- **B** = number of boundary points/edges
- **D** = average depth of triangle following (bounded by mesh diameter, typically O(√M) worst case, but much smaller in practice)

## Initialization Phase (Outside Time Loop)

### 1. Mesh and Setup
- `sample_star_shape`: **O(B)**
- `create_arbitrary_shape_mesh_2d`: **O(M)** - mesh generation complexity
- `assign_positions_arbitrary_2d`: **O(N)**
- `sample_velocities_from_maxwellian_2d`: **O(N)**

### 2. Data Structure Construction
- `create_cell_list_and_adjacency_lists`: **O(M)**
  - Create cells: O(M)
  - Build edge_to_cells: O(M) - each cell has 3 edges
  - Build adjacency_dict: O(M) - each edge shared by at most 2 cells

### 3. Initial Particle Assignment
- Loop through N particles, for each check M cells: **O(N × M)** worst case
- **Typical case**: O(N) with early break when cell found
- **Worst case**: O(N × M) if particles are poorly distributed

**Total Initialization: O(B + M + N × M) = O(N × M)** (assuming N × M dominates)

## Main Time Loop (n_tot iterations)

### Per Iteration Complexity:

#### 1. Collision Algorithm
```python
for cell in cell_list:  # M cells
    num_collisions = cell.num_collisions(dt, e)  # O(1)
    # Process collisions for particles in cell
```
- Loop over M cells: **O(M)**
- For each cell with P_cell particles:
  - `num_collisions`: O(1)
  - Collision processing: O(P_cell) operations
- **Total**: O(M × P_avg) = **O(N)** per iteration
  - Since Σ(P_cell) = N across all cells

#### 2. Boundary Conditions
```python
for cell in cell_list:  # M cells
    reflecting_BC_arbitrary_shape(...)  # O(P_cell × B)
```
- Loop over M cells: **O(M)**
- For each cell, `reflecting_BC_arbitrary_shape`:
  - Point-in-polygon test: O(B) per particle
  - Find closest edge: O(B) per outside particle
- **Total**: O(N × B) per iteration

#### 3. Rebinning Step

**a) Detection Phase:**
```python
for cell in cell_list:  # M cells
    for i, position in enumerate(cell.particle_positions):  # P_cell particles
        if not cell.is_inside(...):  # O(1) - barycentric check
```
- Check all N particles: **O(N)** per iteration

**b) Vectorized Nearest Centroid:**
```python
find_nearest_centroid_cell_vectorized(positions_to_rebin, cell_list)
```
- Create cell_centers array: O(M)
- Compute distances: O(R × M) - R positions × M cells
- Find argmin: O(R × M)
- **Total**: **O(R × M)** per iteration

**c) Triangle Following:**
```python
for each particle to rebin:  # R particles
    find_containing_cell(...)  # O(D) where D is depth
```
- `triangle_to_follow`: O(1) per call (constant time edge lookup)
- `find_containing_cell`: O(D) where D is average depth
  - D is bounded by mesh diameter: O(√M) worst case
  - In practice: D << M (typically D ≈ 5-20 for well-formed meshes)
- **Total**: **O(R × D)** per iteration

**d) Adding Particles:**
- Add R particles to cells: **O(R)** per iteration

**Total Rebinning**: O(N + R × M + R × D + R) = **O(N + R × M)** per iteration
- Typically R << N, so this simplifies to **O(N + R × M)**

## Overall Time Complexity

### Per Time Step:
**O(N + N × B + N + R × M) = O(N × B + R × M)**

### Total Simulation:
**O(N × M + n_tot × (N × B + R × M))**

## Simplified Analysis

### Typical Case Assumptions:
- B ≈ O(√M) (boundary points scale with mesh size)
- R ≈ O(N × dt × v_avg) (particles crossing boundaries per step)
- In practice, R << N (small fraction of particles rebin each step)
- D << M (triangle following depth is small)

### Dominant Terms:
1. **Initialization**: O(N × M) - one-time cost
2. **Per time step**:
   - Collisions: O(N)
   - Boundary conditions: O(N × B) ≈ O(N × √M)
   - Rebinning: O(N + R × M)
     - Detection: O(N)
     - Nearest centroid: O(R × M)
     - Triangle following: O(R × D) ≈ O(R × √M)

### Worst Case (All particles rebin each step, R = N):
**O(N × M + n_tot × (N × B + N × M)) = O(N × M + n_tot × N × (B + M))**

### Best Case (No rebinning, R = 0):
**O(N × M + n_tot × N × B)**

## Space Complexity

- Cells: O(M)
- Particles: O(N)
- Adjacency structures: O(M) - each cell has 2-3 neighbors
- Edge-to-cells mapping: O(M) - 3 edges per cell
- **Total**: **O(N + M)**

## Optimization Notes

1. **Vectorized nearest centroid**: Reduces rebinning from O(R × N × M) to O(R × M)
2. **Triangle following**: Guaranteed to find cell in O(D) steps, avoiding O(M) search
3. **Early break in initial assignment**: Reduces from O(N × M) to O(N) in typical case
4. **Edge-based adjacency**: O(M) instead of O(M²) for adjacency construction

## Practical Performance

For typical simulations:
- N = 10,000 - 1,000,000 particles
- M = 1,000 - 100,000 cells
- n_tot = 100 - 10,000 time steps
- R ≈ 1-10% of N per step (particles crossing boundaries)

**Dominant operations per step:**
- Collisions: O(N) - linear in particles
- Boundary conditions: O(N × B) - can be optimized with spatial indexing
- Rebinning: O(N + R × M) - detection is O(N), lookup is O(R × M)



