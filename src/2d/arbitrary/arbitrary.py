import numpy as np 
import pygmsh 

def create_arbitrary_shape_mesh_2d(N, points, mesh_size=0.1):
    """
    Create an arbitrary shape in 2D using B-spline curves for smooth boundaries.
    
    Parameters
    ----------
    N : int
        Number of particles (not used for mesh generation, kept for compatibility)
    points : array-like
        Boundary points defining the shape
    mesh_size : float, optional
        Characteristic mesh size. Larger values = fewer, larger cells.
        Default is 0.1. Typical range: 0.05 (fine) to 0.5 (coarse).
    
    Returns
    -------
    mesh : pygmsh mesh object
        Generated triangular mesh
    """

    with pygmsh.occ.Geometry() as geom:

        # 1) Add gmsh points with mesh size
        gmpts = [geom.add_point([x, y], mesh_size=mesh_size) for (x, y) in points]

        # 2) Create B-spline curve connecting all points
        # For a closed B-spline, we need to repeat the first point at the end
        closed_points = gmpts + [gmpts[0]]  # Close the curve
        
        # Create B-spline curve with the closed point list
        bspline_curve = geom.add_bspline(closed_points)

        # 3) Close the boundary: use a curve loop, then a plane surface
        loop = geom.add_curve_loop([bspline_curve])
        surf = geom.add_plane_surface(loop)

        # tag the boundary as a Physical Group for later lookup
        boundary_pg = geom.add_physical([bspline_curve], label="boundary")
        domain_pg   = geom.add_physical([surf], label="domain")

        # 4) Generate mesh (triangles by default)
        mesh = geom.generate_mesh()

    return mesh

def assign_positions_arbitrary_2d(N, mesh):

    # 1) Extract triangle data from mesh
    triangle_indices = None
    for cell in mesh.cells:
        if cell.type == "triangle":
            triangle_indices = cell.data
            break
    if triangle_indices is None:
        raise ValueError("No triangles found.")

    mesh_points = mesh.points  # x,y coordinates of all mesh points

    # 2) Calculate triangle areas using cross product
    vertex_a = mesh_points[triangle_indices[:, 0]]  # First vertex of each triangle
    vertex_b = mesh_points[triangle_indices[:, 1]]  # Second vertex of each triangle
    vertex_c = mesh_points[triangle_indices[:, 2]]  # Third vertex of each triangle
    
    # Calculate areas using 2D cross product formula
    triangle_areas = 0.5 * np.abs((vertex_b[:, 0] - vertex_a[:, 0]) * (vertex_c[:, 1] - vertex_a[:, 1]) - 
                                  (vertex_c[:, 0] - vertex_a[:, 0]) * (vertex_b[:, 1] - vertex_a[:, 1]))

    # 3) Choose triangles by area-weighted sampling
    selected_triangle_indices = np.random.choice(len(triangle_indices), size=N, p=triangle_areas / triangle_areas.sum())

    # 4) Generate random barycentric coordinates (Marsaglia trick)
    sqrt_random_1 = np.sqrt(np.random.rand(N, 1))
    random_2 = np.random.rand(N, 1)
    barycentric_u = 1 - sqrt_random_1
    barycentric_v = sqrt_random_1 * (1 - random_2)
    barycentric_w = sqrt_random_1 * random_2

    # 5) Get vertices of selected triangles and compute final positions
    selected_vertex_a = vertex_a[selected_triangle_indices]
    selected_vertex_b = vertex_b[selected_triangle_indices]
    selected_vertex_c = vertex_c[selected_triangle_indices]
    particle_positions = barycentric_u * selected_vertex_a + barycentric_v * selected_vertex_b + barycentric_w * selected_vertex_c
    
    # Ensure we only return 2D positions
    particle_positions = particle_positions[:, :2]
    
    return particle_positions