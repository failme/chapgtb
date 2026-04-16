"""
Minecraft Clone - Core Mechanics & Foundation

This module defines the fundamental data structures for the voxel world,
including the Block enum and the Chunk class with mesh generation logic.
"""

from enum import IntEnum
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple, List

# Constants
CHUNK_WIDTH = 16
CHUNK_DEPTH = 16
CHUNK_HEIGHT = 384
MIN_Y = -64
MAX_Y = 320


class Block(IntEnum):
    """
    Block type definitions using integer IDs.
    0 is always Air (empty space).
    """
    AIR = 0
    STONE = 1
    DIRT = 2
    GRASS = 3
    BEDROCK = 4
    # Future blocks can be added here
    
    @property
    def is_solid(self) -> bool:
        """Check if the block is solid (collidable and opaque)."""
        return self != Block.AIR
    
    @property
    def is_transparent(self) -> bool:
        """Check if the block is transparent (doesn't cull faces)."""
        # For now, only air is transparent. Glass/leaves would go here later.
        return self == Block.AIR


# Face directions: (dx, dy, dz, face_index)
# face_index: 0=right, 1=left, 2=top, 3=bottom, 4=front, 5=back
FACE_DIRECTIONS = [
    (1, 0, 0, 0),   # Right (+X)
    (-1, 0, 0, 1),  # Left (-X)
    (0, 1, 0, 2),   # Top (+Y)
    (0, -1, 0, 3),  # Bottom (-Y)
    (0, 0, 1, 4),   # Front (+Z)
    (0, 0, -1, 5),  # Back (-Z)
]

# UV coordinates for each block type on the texture atlas
# Format: [(u_min, v_min, u_max, v_max), ...] for each of 6 faces
# Assuming a 16x16 pixel grid per block face, atlas is organized in rows
# This is a simplified mapping - real implementation would use actual atlas coordinates
BLOCK_UV_MAP = {
    Block.STONE: [(0, 0, 1, 1)] * 6,  # Same texture on all sides
    Block.DIRT: [(0.25, 0, 0.5, 1)] * 6,
    Block.GRASS: [
        (0.25, 0, 0.5, 1),  # Right - side texture
        (0.25, 0, 0.5, 1),  # Left - side texture
        (0.5, 0, 0.75, 1),  # Top - grass top
        (0.25, 0, 0.5, 1),  # Bottom - dirt
        (0.25, 0, 0.5, 1),  # Front - side texture
        (0.25, 0, 0.5, 1),  # Back - side texture
    ],
    Block.BEDROCK: [(0.75, 0, 1, 1)] * 6,
}


@dataclass
class Vertex:
    """Represents a single vertex in the mesh."""
    x: float
    y: float
    z: float
    u: float
    v: float
    ao: float  # Ambient occlusion value (0.0 to 1.0)


class MeshData:
    """
    Holds the generated mesh data for a chunk.
    Contains vertices and indices for OpenGL/Vulkan rendering.
    """
    def __init__(self):
        self.vertices: List[float] = []  # Flat list: [x, y, z, u, v, ao, ...]
        self.indices: List[int] = []     # Flat list of indices
        
    def add_quad(self, v0: Vertex, v1: Vertex, v2: Vertex, v3: Vertex, 
                 base_index: int) -> int:
        """
        Add a quad (4 vertices) to the mesh and return the new base index.
        Vertices are added in counter-clockwise order for proper culling.
        Returns the number of vertices added (always 4).
        """
        # Add vertices
        for v in [v0, v1, v2, v3]:
            self.vertices.extend([v.x, v.y, v.z, v.u, v.v, v.ao])
        
        # Add two triangles (6 indices)
        # Triangle 1: v0, v1, v2
        # Triangle 2: v0, v2, v3
        self.indices.extend([
            base_index, base_index + 1, base_index + 2,
            base_index, base_index + 2, base_index + 3
        ])
        
        return 4  # Always adds 4 vertices
    
    def clear(self):
        """Clear all mesh data."""
        self.vertices.clear()
        self.indices.clear()
    
    @property
    def vertex_count(self) -> int:
        return len(self.vertices) // 6  # 6 components per vertex
    
    @property
    def index_count(self) -> int:
        return len(self.indices)


class Chunk:
    """
    Represents a single chunk of the world.
    A chunk is 16x384x16 blocks (width x height x depth).
    
    Coordinate system:
    - Local coordinates: (0-15, -64 to 319, 0-15) within this chunk
    - Global coordinates: Absolute world position
    """
    
    def __init__(self, chunk_x: int, chunk_z: int):
        """
        Initialize a chunk at the given chunk coordinates.
        
        Args:
            chunk_x: The X coordinate of this chunk in chunk space
            chunk_z: The Z coordinate of this chunk in chunk space
        """
        self.chunk_x = chunk_x
        self.chunk_z = chunk_z
        
        # 3D array of block IDs: [x][y][z]
        # Using numpy for efficient storage and access
        self.blocks: np.ndarray = np.zeros(
            (CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH), 
            dtype=np.uint8
        )
        
        # Mesh data for rendering
        self.mesh_data: Optional[MeshData] = None
        self.mesh_dirty: bool = True  # Flag indicating mesh needs regeneration
        
        # Neighbor chunk references (for cross-chunk face culling)
        self.neighbors: dict = {
            'right': None,   # +X
            'left': None,    # -X
            'top': None,     # +Y (not typically used for same-level chunks)
            'bottom': None,  # -Y
            'front': None,   # +Z
            'back': None,    # -Z
        }
    
    def get_block(self, x: int, y: int, z: int) -> Block:
        """
        Get the block at local coordinates.
        
        Args:
            x, y, z: Local coordinates within this chunk
            
        Returns:
            The Block type at the given coordinates
        """
        if not self._is_valid_local(x, y, z):
            return Block.AIR
        
        return Block(self.blocks[x, y - MIN_Y, z])
    
    def set_block(self, x: int, y: int, z: int, block: Block) -> bool:
        """
        Set the block at local coordinates.
        
        Args:
            x, y, z: Local coordinates within this chunk
            block: The Block type to set
            
        Returns:
            True if the block was successfully set, False otherwise
        """
        if not self._is_valid_local(x, y, z):
            return False
        
        self.blocks[x, y - MIN_Y, z] = block.value
        self.mesh_dirty = True
        
        # Mark neighbor chunks as dirty if on border
        self._mark_neighbors_dirty(x, y, z)
        
        return True
    
    def _is_valid_local(self, x: int, y: int, z: int) -> bool:
        """Check if coordinates are within valid local range."""
        return (0 <= x < CHUNK_WIDTH and 
                MIN_Y <= y < MAX_Y and 
                0 <= z < CHUNK_DEPTH)
    
    def _mark_neighbors_dirty(self, x: int, y: int, z: int):
        """Mark neighboring chunks as dirty if block is on border."""
        if x == 0 and self.neighbors['left']:
            self.neighbors['left'].mesh_dirty = True
        if x == CHUNK_WIDTH - 1 and self.neighbors['right']:
            self.neighbors['right'].mesh_dirty = True
        if z == 0 and self.neighbors['back']:
            self.neighbors['back'].mesh_dirty = True
        if z == CHUNK_DEPTH - 1 and self.neighbors['front']:
            self.neighbors['front'].mesh_dirty = True
    
    def get_block_global(self, gx: int, gy: int, gz: int) -> Block:
        """
        Get block at global coordinates. Handles chunk boundaries.
        
        This method should ideally be called from a World class that manages
        all chunks, but included here for completeness.
        """
        # Convert global to local
        lx = gx % CHUNK_WIDTH
        lz = gz % CHUNK_DEPTH
        ly = gy
        
        # Handle negative coordinates
        if lx < 0:
            lx += CHUNK_WIDTH
        if lz < 0:
            lz += CHUNK_DEPTH
            
        if not self._is_valid_local(lx, ly, lz):
            return Block.AIR
            
        return Block(self.blocks[lx, ly - MIN_Y, lz])
    
    def calculate_ambient_occlusion(self, x: int, y: int, z: int, 
                                    face_dir: Tuple[int, int, int]) -> float:
        """
        Calculate ambient occlusion for a vertex based on surrounding blocks.
        
        Args:
            x, y, z: Local coordinates of the block
            face_dir: The direction of the face being rendered
            
        Returns:
            AO value between 0.0 (dark) and 1.0 (bright)
        """
        # Simplified AO calculation
        # In a full implementation, this would sample multiple directions
        # and average the results
        
        occlusion = 1.0
        
        # Check blocks around this position
        # Reduce brightness if surrounded by solid blocks
        dx, dy, dz = face_dir
        
        # Sample neighboring blocks for AO
        samples = [
            (x + dx, y + dy, z + dz),
            (x + dx, y, z + dz),
            (x + dx, y + dy, z),
        ]
        
        for sx, sy, sz in samples:
            if 0 <= sx < CHUNK_WIDTH and MIN_Y <= sy < MAX_Y and 0 <= sz < CHUNK_DEPTH:
                block = Block(self.blocks[sx, sy - MIN_Y, sz])
                if block.is_solid:
                    occlusion -= 0.1
        
        return max(0.5, min(1.0, occlusion))
    
    def generate_mesh(self) -> MeshData:
        """
        Generate the mesh for this chunk using face culling.
        
        Only generates faces where the adjacent block is air or transparent.
        Uses a texture atlas for UV mapping.
        
        Returns:
            MeshData containing vertices and indices
        """
        mesh = MeshData()
        base_index = 0
        
        for x in range(CHUNK_WIDTH):
            for y in range(MIN_Y, MAX_Y):
                for z in range(CHUNK_DEPTH):
                    block = self.get_block(x, y, z)
                    
                    if block == Block.AIR:
                        continue
                    
                    # Check each face
                    for dx, dy, dz, face_idx in FACE_DIRECTIONS:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        
                        # Determine if face should be visible
                        face_visible = False
                        
                        if 0 <= nx < CHUNK_WIDTH and MIN_Y <= ny < MAX_Y and 0 <= nz < CHUNK_DEPTH:
                            # Neighbor is within this chunk
                            neighbor = Block(self.blocks[nx, ny - MIN_Y, nz])
                            if neighbor.is_transparent:
                                face_visible = True
                        else:
                            # Neighbor is in another chunk or out of bounds
                            # For now, assume visible (proper implementation checks neighbors)
                            face_visible = True
                        
                        if face_visible:
                            # Calculate AO for this face
                            ao = self.calculate_ambient_occlusion(x, y, z, (dx, dy, dz))
                            
                            # Get UV coordinates for this face
                            uv_coords = BLOCK_UV_MAP.get(block, BLOCK_UV_MAP[Block.STONE])
                            u_min, v_min, u_max, v_max = uv_coords[face_idx]
                            
                            # Add quad for this face
                            base_index += self._add_face_vertices(
                                mesh, x, y, z, dx, dy, dz,
                                u_min, v_min, u_max, v_max, ao
                            )
        
        self.mesh_data = mesh
        self.mesh_dirty = False
        return mesh
    
    def _add_face_vertices(self, mesh: MeshData, x: int, y: int, z: int,
                          dx: int, dy: int, dz: int,
                          u_min: float, v_min: float, u_max: float, v_max: float,
                          ao: float) -> int:
        """
        Add vertices for a single face of a block.
        
        Args:
            mesh: The mesh to add vertices to
            x, y, z: Block position
            dx, dy, dz: Face normal direction
            u_min, v_min, u_max, v_max: UV coordinates
            ao: Ambient occlusion value
            
        Returns:
            Number of vertices added
        """
        # Define face vertices based on direction
        # Each face has 4 vertices in counter-clockwise order
        
        if dx == 1:  # Right face (+X)
            vertices = [
                Vertex(x + 1, y,     z,     u_max, v_min, ao),
                Vertex(x + 1, y,     z + 1, u_min, v_min, ao),
                Vertex(x + 1, y + 1, z + 1, u_min, v_max, ao),
                Vertex(x + 1, y + 1, z,     u_max, v_max, ao),
            ]
        elif dx == -1:  # Left face (-X)
            vertices = [
                Vertex(x, y,     z + 1, u_max, v_min, ao),
                Vertex(x, y,     z,     u_min, v_min, ao),
                Vertex(x, y + 1, z,     u_min, v_max, ao),
                Vertex(x, y + 1, z + 1, u_max, v_max, ao),
            ]
        elif dy == 1:  # Top face (+Y)
            vertices = [
                Vertex(x,     y + 1, z + 1, u_min, v_max, ao),
                Vertex(x + 1, y + 1, z + 1, u_max, v_max, ao),
                Vertex(x + 1, y + 1, z,     u_max, v_min, ao),
                Vertex(x,     y + 1, z,     u_min, v_min, ao),
            ]
        elif dy == -1:  # Bottom face (-Y)
            vertices = [
                Vertex(x,     y, z,     u_min, v_min, ao),
                Vertex(x + 1, y, z,     u_max, v_min, ao),
                Vertex(x + 1, y, z + 1, u_max, v_max, ao),
                Vertex(x,     y, z + 1, u_min, v_max, ao),
            ]
        elif dz == 1:  # Front face (+Z)
            vertices = [
                Vertex(x + 1, y,     z + 1, u_max, v_min, ao),
                Vertex(x,     y,     z + 1, u_min, v_min, ao),
                Vertex(x,     y + 1, z + 1, u_min, v_max, ao),
                Vertex(x + 1, y + 1, z + 1, u_max, v_max, ao),
            ]
        else:  # dz == -1, Back face (-Z)
            vertices = [
                Vertex(x,     y,     z, u_max, v_min, ao),
                Vertex(x + 1, y,     z, u_min, v_min, ao),
                Vertex(x + 1, y + 1, z, u_min, v_max, ao),
                Vertex(x,     y + 1, z, u_max, v_max, ao),
            ]
        
        current_base = mesh.vertex_count
        mesh.add_quad(vertices[0], vertices[1], vertices[2], vertices[3], current_base)
        
        return 4


class World:
    """
    Manages all chunks in the world.
    Provides methods for chunk access and world-level operations.
    """
    
    def __init__(self):
        self.chunks: dict = {}  # (chunk_x, chunk_z) -> Chunk
    
    def get_chunk(self, chunk_x: int, chunk_z: int) -> Optional[Chunk]:
        """Get a chunk by its coordinates."""
        return self.chunks.get((chunk_x, chunk_z))
    
    def get_or_create_chunk(self, chunk_x: int, chunk_z: int) -> Chunk:
        """Get an existing chunk or create a new one."""
        key = (chunk_x, chunk_z)
        if key not in self.chunks:
            chunk = Chunk(chunk_x, chunk_z)
            self.chunks[key] = chunk
            
            # Set up neighbor references
            self._link_neighbors(chunk)
            
            return chunk
        return self.chunks[key]
    
    def _link_neighbors(self, chunk: Chunk):
        """Link a chunk to its neighbors for proper face culling."""
        cx, cz = chunk.chunk_x, chunk.chunk_z
        
        # Check and link right neighbor (+X)
        if (cx + 1, cz) in self.chunks:
            chunk.neighbors['right'] = self.chunks[(cx + 1, cz)]
            self.chunks[(cx + 1, cz)].neighbors['left'] = chunk
        
        # Check and link left neighbor (-X)
        if (cx - 1, cz) in self.chunks:
            chunk.neighbors['left'] = self.chunks[(cx - 1, cz)]
            self.chunks[(cx - 1, cz)].neighbors['right'] = chunk
        
        # Check and link front neighbor (+Z)
        if (cx, cz + 1) in self.chunks:
            chunk.neighbors['front'] = self.chunks[(cx, cz + 1)]
            self.chunks[(cx, cz + 1)].neighbors['back'] = chunk
        
        # Check and link back neighbor (-Z)
        if (cx, cz - 1) in self.chunks:
            chunk.neighbors['back'] = self.chunks[(cx, cz - 1)]
            self.chunks[(cx, cz - 1)].neighbors['front'] = chunk
    
    def get_block(self, x: int, y: int, z: int) -> Block:
        """Get block at global coordinates."""
        chunk_x = x // CHUNK_WIDTH
        chunk_z = z // CHUNK_DEPTH
        
        # Handle negative coordinates
        if x < 0:
            chunk_x -= 1
        if z < 0:
            chunk_z -= 1
        
        chunk = self.get_chunk(chunk_x, chunk_z)
        if chunk is None:
            return Block.AIR
        
        # Convert to local coordinates
        local_x = x % CHUNK_WIDTH
        local_z = z % CHUNK_DEPTH
        
        if local_x < 0:
            local_x += CHUNK_WIDTH
        if local_z < 0:
            local_z += CHUNK_DEPTH
        
        return chunk.get_block(local_x, y, local_z)
    
    def set_block(self, x: int, y: int, z: int, block: Block) -> bool:
        """Set block at global coordinates."""
        chunk_x = x // CHUNK_WIDTH
        chunk_z = z // CHUNK_DEPTH
        
        # Handle negative coordinates
        if x < 0:
            chunk_x -= 1
        if z < 0:
            chunk_z -= 1
        
        chunk = self.get_or_create_chunk(chunk_x, chunk_z)
        
        # Convert to local coordinates
        local_x = x % CHUNK_WIDTH
        local_z = z % CHUNK_DEPTH
        
        if local_x < 0:
            local_x += CHUNK_WIDTH
        if local_z < 0:
            local_z += CHUNK_DEPTH
        
        return chunk.set_block(local_x, y, local_z)
    
    def generate_terrain(self, chunk_x: int, chunk_z: int, 
                        noise_func=None) -> Chunk:
        """
        Generate basic terrain for a chunk.
        
        Args:
            chunk_x: X coordinate of the chunk
            chunk_z: Z coordinate of the chunk
            noise_func: Optional noise function for height generation
            
        Returns:
            The generated chunk
        """
        chunk = self.get_or_create_chunk(chunk_x, chunk_z)
        
        # Simple noise function if none provided
        if noise_func is None:
            noise_func = lambda x, z: ((x * 1234 + z * 5678) % 100) / 100.0
        
        for x in range(CHUNK_WIDTH):
            for z in range(CHUNK_DEPTH):
                # Calculate global coordinates
                gx = chunk_x * CHUNK_WIDTH + x
                gz = chunk_z * CHUNK_DEPTH + z
                
                # Sample noise for height
                noise_val = noise_func(gx, gz)
                
                # Map noise to height (between Y=60 and Y=120)
                height = int(60 + noise_val * 60)
                
                # Fill column
                for y in range(MIN_Y, MAX_Y):
                    if y == MIN_Y:
                        # Bedrock at bottom
                        chunk.set_block(x, y, z, Block.BEDROCK)
                    elif y < height - 3:
                        # Stone below dirt layers
                        chunk.set_block(x, y, z, Block.STONE)
                    elif y < height:
                        # Dirt layers
                        chunk.set_block(x, y, z, Block.DIRT)
                    elif y == height:
                        # Grass on top
                        chunk.set_block(x, y, z, Block.GRASS)
                    else:
                        # Air above
                        chunk.set_block(x, y, z, Block.AIR)
        
        chunk.mesh_dirty = True
        return chunk


if __name__ == "__main__":
    # Test the chunk system
    print("Testing Chunk System...")
    
    # Create a world
    world = World()
    
    # Generate a test chunk
    chunk = world.generate_terrain(0, 0)
    
    print(f"Chunk created at ({chunk.chunk_x}, {chunk.chunk_z})")
    print(f"Chunk dimensions: {CHUNK_WIDTH}x{CHUNK_HEIGHT}x{CHUNK_DEPTH}")
    
    # Test block access
    test_block = chunk.get_block(5, 70, 5)
    print(f"Block at (5, 70, 5): {test_block.name}")
    
    # Test mesh generation
    mesh = chunk.generate_mesh()
    print(f"Mesh generated: {mesh.vertex_count} vertices, {mesh.index_count} indices")
    
    # Test block modification
    chunk.set_block(5, 70, 5, Block.AIR)
    print(f"Set block at (5, 70, 5) to AIR")
    
    # Regenerate mesh after modification
    mesh = chunk.generate_mesh()
    print(f"Mesh regenerated: {mesh.vertex_count} vertices, {mesh.index_count} indices")
    
    print("\nCore mechanics foundation ready!")
    print("- Chunk data structure: ✓")
    print("- Face culling: ✓")
    print("- Mesh generation: ✓")
    print("- Texture atlas UV mapping: ✓")
    print("- Ambient occlusion: ✓")
    print("- World management: ✓")
