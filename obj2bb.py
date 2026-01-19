
import json
import os
import sys
import logging
from pathlib import Path
import struct
from typing import List, Tuple, Set, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

FACE_THICKNESS = 0.01  # Default thickness for face cubes
MIN_CUBE_SIZE = 0.05   # Minimum size for any cube on any axis
MAX_CUBE_COUNT = None  # Default: no limit
MAX_FACE_SIZE = None   # Default: no limit on face size
ROUNDING_PRECISION = 5  # Decimal places for rounding coordinates

def validate_parameters(thickness: float, min_cube_size: float, max_cube_count: Optional[int], max_face_size: Optional[float]) -> None:
    """Validate input parameters."""
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")
    if min_cube_size <= 0:
        raise ValueError(f"min_cube_size must be positive, got {min_cube_size}")
    if max_cube_count is not None and max_cube_count <= 0:
        raise ValueError(f"max_cube_count must be positive, got {max_cube_count}")
    if max_face_size is not None and max_face_size <= 0:
        raise ValueError(f"max_face_size must be positive, got {max_face_size}")

def get_unique_output_path(input_path: str) -> str:
    """Generate a unique output filename based on input filename.
    
    Args:
        input_path: Path to the input file
        
    Returns:
        A unique output filepath that doesn't already exist
    """
    input_name = Path(input_path).stem
    base_output = f"{input_name}.geo.json"
    
    if not os.path.exists(base_output):
        return base_output
    
    counter = 1
    while True:
        unique_output = f"{input_name}_{counter}.geo.json"
        if not os.path.exists(unique_output):
            return unique_output
        counter += 1

def calculate_cube_importance(origin: List[float], size: List[float]) -> float:
    """Calculate cube importance based on volume and position.
    
    Args:
        origin: [x, y, z] coordinates of cube origin
        size: [width, height, depth] of cube
        
    Returns:
        Importance score (higher = more important)
    """
    volume = size[0] * size[1] * size[2]
    # Cubes closer to origin are slightly more important
    distance_from_origin = (origin[0]**2 + origin[1]**2 + origin[2]**2)**0.5
    # Prioritize volume over position (adjust weights as needed)
    importance = volume * 100 - distance_from_origin
    return importance

def filter_cubes_by_importance(cubes: List[Dict], max_count: Optional[int]) -> List[Dict]:
    """Filter cubes to keep only the most important ones up to max_count.
    
    Args:
        cubes: List of cube dictionaries with 'origin' and 'size' keys
        max_count: Maximum number of cubes to keep (None = no limit)
        
    Returns:
        Filtered list of cubes
    """
    if max_count is None or len(cubes) <= max_count:
        return cubes
    
    # Calculate importance for each cube
    cube_importance = []
    for i, cube in enumerate(cubes):
        importance = calculate_cube_importance(cube["origin"], cube["size"])
        cube_importance.append((importance, i, cube))
    
    # Sort by importance (descending) and take top max_count
    cube_importance.sort(key=lambda x: x[0], reverse=True)
    filtered_cubes = [cube for _, _, cube in cube_importance[:max_count]]
    
    logger.info(f"Filtered from {len(cubes)} to {len(filtered_cubes)} cubes based on importance")
    return filtered_cubes

def vertices_are_connected(face_vertices: List[Tuple[float, float, float]]) -> bool:
    """Check if face vertices form a connected polygon (no degenerate faces).
    
    Verifies that:
    1. No vertices are duplicated
    2. Vertices form a continuous path (no isolated vertices)
    
    Args:
        face_vertices: List of 3D vertices for the face
        
    Returns:
        True if vertices are properly connected, False otherwise
    """
    if len(face_vertices) < 3:
        return False
    
    # Check for duplicate vertices
    unique_verts = set(face_vertices)
    if len(unique_verts) < 3:
        # Degenerate face with duplicate vertices
        return False
    
    # Check if all vertices have at least 2 edges (not isolated)
    # In a proper polygon, every vertex should connect to at least 2 others
    for vertex in face_vertices:
        # Count occurrences - each vertex should appear at least once in a valid face
        if face_vertices.count(vertex) == 0:
            return False
    
    return True

def process_face_vertices(face_vertices: List[Tuple[float, float, float]], 
                          thickness: float, 
                          min_cube_size: float,
                          max_face_size: Optional[float],
                          unique_cubes: Set[Tuple]) -> Optional[Dict]:
    """Process a single face and return a cube if valid.
    
    Args:
        face_vertices: List of 3D vertices for the face
        thickness: Thickness to apply to flat faces
        min_cube_size: Minimum allowed cube size
        max_face_size: Maximum allowed face size (None = no limit)
        unique_cubes: Set of existing cube signatures for deduplication
        
    Returns:
        A cube dictionary or None if the face doesn't meet criteria
    """
    if not face_vertices or len(face_vertices) < 3:
        return None
    
    # Only process faces with connected vertices
    if not vertices_are_connected(face_vertices):
        return None
        
    x_coords, y_coords, z_coords = zip(*face_vertices)
    
    # Find the axis with the smallest range (the "flat" axis)
    ranges = [
        max(x_coords) - min(x_coords),
        max(y_coords) - min(y_coords),
        max(z_coords) - min(z_coords)
    ]
    flat_axis = ranges.index(min(ranges))
    
    origin = [
        min(x_coords),
        min(y_coords),
        min(z_coords)
    ]
    size = [
        max(x_coords) - min(x_coords),
        max(y_coords) - min(y_coords),
        max(z_coords) - min(z_coords)
    ]
    
    # If the face is perfectly flat, give it thickness
    if size[flat_axis] < thickness:
        origin[flat_axis] -= thickness / 2
        size[flat_axis] = thickness
    
    # Skip cubes that are still too thin after adjustment
    if size[flat_axis] < min_cube_size:
        return None
    
    # Skip cubes that exceed maximum face size
    if max_face_size is not None and any(s > max_face_size for s in size):
        return None
    
    # Round values for comparison
    rounded_origin = tuple(round(o, ROUNDING_PRECISION) for o in origin)
    rounded_size = tuple(round(s, ROUNDING_PRECISION) for s in size)
    cube_signature = (rounded_origin, rounded_size)
    
    # Skip if this cube already exists
    if cube_signature in unique_cubes:
        return None
    
    unique_cubes.add(cube_signature)
    
    return {
        "origin": list(rounded_origin),
        "size": list(rounded_size),
        "uv": [0, 0]
    }

def create_bedrock_model(model_name: str, cubes: List[Dict], texture_width: int = 64, texture_height: int = 64) -> Dict:
    """Create a Bedrock entity model structure.
    
    Args:
        model_name: Name of the geometry
        cubes: List of cube dictionaries
        texture_width: Texture width in pixels
        texture_height: Texture height in pixels
        
    Returns:
        Complete Bedrock model dictionary
    """
    return {
        f"geometry.{model_name}": {
            "texturewidth": texture_width,
            "textureheight": texture_height,
            "bones": [
                {
                    "name": "head",
                    "pivot": [0, 0, 0],
                    "rotation": [0, 0, 0],
                    "cubes": cubes
                }
            ]
        }
    }

def save_model_to_file(bedrock_model: Dict, output_path: str) -> None:
    """Save Bedrock model to JSON file.
    
    Args:
        bedrock_model: The model dictionary to save
        output_path: Output file path
    """
    try:
        with open(output_path, "w") as json_file:
            json.dump(bedrock_model, json_file, indent=4)
        logger.info(f"Bedrock model saved to {output_path}")
    except IOError as e:
        logger.error(f"Failed to write output file: {e}")
        raise

def obj_to_bedrock_entity(obj_path: str, texture_width: int = 64, texture_height: int = 64, 
                          thickness: float = FACE_THICKNESS, min_cube_size: float = MIN_CUBE_SIZE, 
                          max_cube_count: Optional[int] = MAX_CUBE_COUNT, 
                          max_face_size: Optional[float] = MAX_FACE_SIZE) -> None:
    """Convert OBJ file to Bedrock entity.
    
    Args:
        obj_path: Path to OBJ file
        texture_width: Texture width in pixels
        texture_height: Texture height in pixels
        thickness: Thickness for flat faces
        min_cube_size: Minimum cube size
        max_cube_count: Maximum number of cubes to keep
        max_face_size: Maximum face size
    """
    validate_parameters(thickness, min_cube_size, max_cube_count, max_face_size)
    
    obj_name = Path(obj_path).stem
    unique_cubes: Set[Tuple] = set()
    temp_cubes: List[Dict] = []
    
    try:
        with open(obj_path, "r") as obj_file:
            vertices: List[Tuple[float, float, float]] = []
            faces: List[List[int]] = []
            
            for line in obj_file:
                line = line.strip()
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        x, y, z = map(float, parts[1:4])
                        vertices.append((x, y, z))
                elif line.startswith("f "):
                    indices = []
                    for part in line.split()[1:]:
                        idx = part.split('/')[0]
                        if idx:
                            indices.append(int(idx) - 1)
                    if len(indices) >= 3:
                        faces.append(indices)
            
            for face in faces:
                # Validate face indices
                if any(idx >= len(vertices) or idx < 0 for idx in face):
                    logger.warning(f"Skipping face with invalid vertex indices: {face}")
                    continue
                    
                face_vertices = [vertices[i] for i in face]
                cube = process_face_vertices(face_vertices, thickness, min_cube_size, 
                                            max_face_size, unique_cubes)
                if cube:
                    temp_cubes.append(cube)
    
    except FileNotFoundError:
        logger.error(f"OBJ file not found: {obj_path}")
        raise
    except ValueError as e:
        logger.error(f"Error parsing OBJ file: {e}")
        raise
    
    # Filter cubes by importance if max_cube_count is specified
    filtered_cubes = filter_cubes_by_importance(temp_cubes, max_cube_count)
    bedrock_model = create_bedrock_model(obj_name, filtered_cubes, texture_width, texture_height)
    
    output_path = get_unique_output_path(obj_path)
    save_model_to_file(bedrock_model, output_path)

def stl_to_bedrock_entity(stl_path: str, texture_width: int = 64, texture_height: int = 64, 
                          thickness: float = FACE_THICKNESS, min_cube_size: float = MIN_CUBE_SIZE,
                          max_cube_count: Optional[int] = MAX_CUBE_COUNT, 
                          max_face_size: Optional[float] = MAX_FACE_SIZE) -> None:
    """Convert STL file to Bedrock entity.
    
    Args:
        stl_path: Path to STL file
        texture_width: Texture width in pixels
        texture_height: Texture height in pixels
        thickness: Thickness for flat faces
        min_cube_size: Minimum cube size
        max_cube_count: Maximum number of cubes to keep
        max_face_size: Maximum face size
    """
    validate_parameters(thickness, min_cube_size, max_cube_count, max_face_size)
    
    stl_name = Path(stl_path).stem
    unique_cubes: Set[Tuple] = set()
    temp_cubes: List[Dict] = []
    
    try:
        with open(stl_path, "rb") as stl_file:
            header = stl_file.read(80)
            num_triangles_bytes = stl_file.read(4)
            if len(num_triangles_bytes) < 4:
                raise ValueError("Invalid STL file: cannot read triangle count")
                
            num_triangles = struct.unpack('<I', num_triangles_bytes)[0]
            
            for triangle_idx in range(num_triangles):
                try:
                    stl_file.read(12)  # Normal vector, not used
                    vertices_data = stl_file.read(36)  # 3 vertices * 12 bytes each
                    if len(vertices_data) < 36:
                        logger.warning(f"Incomplete triangle data at index {triangle_idx}")
                        break
                    
                    vertices = [struct.unpack('<fff', vertices_data[i*12:(i+1)*12]) for i in range(3)]
                    cube = process_face_vertices(vertices, thickness, min_cube_size, 
                                                max_face_size, unique_cubes)
                    if cube:
                        temp_cubes.append(cube)
                    
                    stl_file.read(2)  # Attribute byte count
                except struct.error as e:
                    logger.warning(f"Error parsing triangle {triangle_idx}: {e}")
                    continue
    
    except FileNotFoundError:
        logger.error(f"STL file not found: {stl_path}")
        raise
    except struct.error as e:
        logger.error(f"Error parsing STL file: {e}")
        raise
    
    # Filter cubes by importance if max_cube_count is specified
    filtered_cubes = filter_cubes_by_importance(temp_cubes, max_cube_count)
    bedrock_model = create_bedrock_model(stl_name, filtered_cubes, texture_width, texture_height)
    
    output_path = get_unique_output_path(stl_path)
    save_model_to_file(bedrock_model, output_path)

def ply_to_bedrock_entity(ply_path: str, texture_width: int = 64, texture_height: int = 64, 
                          thickness: float = FACE_THICKNESS, min_cube_size: float = MIN_CUBE_SIZE,
                          max_cube_count: Optional[int] = MAX_CUBE_COUNT, 
                          max_face_size: Optional[float] = MAX_FACE_SIZE) -> None:
    """Convert PLY file to Bedrock entity.
    
    Args:
        ply_path: Path to PLY file
        texture_width: Texture width in pixels
        texture_height: Texture height in pixels
        thickness: Thickness for flat faces
        min_cube_size: Minimum cube size
        max_cube_count: Maximum number of cubes to keep
        max_face_size: Maximum face size
    """
    validate_parameters(thickness, min_cube_size, max_cube_count, max_face_size)
    
    ply_name = Path(ply_path).stem
    unique_cubes: Set[Tuple] = set()
    temp_cubes: List[Dict] = []
    
    try:
        with open(ply_path, "r") as ply_file:
            vertices: List[Tuple[float, float, float]] = []
            faces: List[List[int]] = []
            header = True
            vertex_count = 0
            face_count = 0
            line_index = 0
            
            for line in ply_file:
                line = line.strip()
                if header:
                    if line.startswith("element vertex"):
                        vertex_count = int(line.split()[-1])
                    elif line.startswith("element face"):
                        face_count = int(line.split()[-1])
                    elif line.startswith("end_header"):
                        header = False
                    continue
                
                if line_index < vertex_count:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            x, y, z = map(float, parts[:3])
                            vertices.append((x, y, z))
                        except ValueError as e:
                            logger.warning(f"Skipping malformed vertex at line {line_index}: {e}")
                    line_index += 1
                elif line_index < vertex_count + face_count:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            indices = list(map(int, parts[1:]))
                            if len(indices) >= 3:
                                # Validate indices
                                if any(idx >= len(vertices) or idx < 0 for idx in indices):
                                    logger.warning(f"Skipping face with invalid vertex indices: {indices}")
                                else:
                                    faces.append(indices)
                        except ValueError as e:
                            logger.warning(f"Skipping malformed face at line {line_index}: {e}")
                    line_index += 1
            
            for face in faces:
                face_vertices = [vertices[i] for i in face]
                cube = process_face_vertices(face_vertices, thickness, min_cube_size, 
                                            max_face_size, unique_cubes)
                if cube:
                    temp_cubes.append(cube)
    
    except FileNotFoundError:
        logger.error(f"PLY file not found: {ply_path}")
        raise
    except Exception as e:
        logger.error(f"Error parsing PLY file: {e}")
        raise
    
    # Filter cubes by importance if max_cube_count is specified
    filtered_cubes = filter_cubes_by_importance(temp_cubes, max_cube_count)
    bedrock_model = create_bedrock_model(ply_name, filtered_cubes, texture_width, texture_height)
    
    output_path = get_unique_output_path(ply_path)
    save_model_to_file(bedrock_model, output_path)

def main() -> None:
    """Main entry point for the script."""
    try:
        if len(sys.argv) < 2:
            print(f"Usage: python {os.path.basename(__file__)} <ModelFilePath> [FACE_THICKNESS] [MIN_CUBE_SIZE] [MAX_CUBE_COUNT] [MAX_FACE_SIZE]")
            print("\nSupported formats: OBJ, STL, PLY")
            sys.exit(1)
        
        model_path = str(sys.argv[1]).replace("\\", "/")
        
        # Validate file exists
        if not os.path.exists(model_path):
            logger.error(f"File not found: {model_path}")
            sys.exit(1)
        
        # Optional: thickness, min_cube_size, max_cube_count, and max_face_size from command line
        try:
            thickness = float(sys.argv[2]) if len(sys.argv) > 2 else FACE_THICKNESS
            min_cube_size = float(sys.argv[3]) if len(sys.argv) > 3 else MIN_CUBE_SIZE
            max_cube_count = int(sys.argv[4]) if len(sys.argv) > 4 else MAX_CUBE_COUNT
            max_face_size = float(sys.argv[5]) if len(sys.argv) > 5 else MAX_FACE_SIZE
        except ValueError as e:
            logger.error(f"Invalid parameter format: {e}")
            sys.exit(1)
        
        file_extension = Path(model_path).suffix.lower()
        logger.info(f"Processing {file_extension} file: {model_path}")
        
        if file_extension == ".obj":
            obj_to_bedrock_entity(model_path, thickness=thickness, min_cube_size=min_cube_size, 
                                 max_cube_count=max_cube_count, max_face_size=max_face_size)
        elif file_extension == ".stl":
            stl_to_bedrock_entity(model_path, thickness=thickness, min_cube_size=min_cube_size,
                                 max_cube_count=max_cube_count, max_face_size=max_face_size)
        elif file_extension == ".ply":
            ply_to_bedrock_entity(model_path, thickness=thickness, min_cube_size=min_cube_size,
                                 max_cube_count=max_cube_count, max_face_size=max_face_size)
        else:
            logger.error("Unsupported file format. Supported formats are: OBJ, STL, PLY.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()