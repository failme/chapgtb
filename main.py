"""
Minecraft Clone - Phases 1-4 Complete (OPTIMIZED)
Run with: python main.py
Controls: WASD Move, Space Jump, Click Break/Place, E Inventory, 1-9 Hotbar

PERFORMANCE OPTIMIZATIONS APPLIED:
1. Merged mesh rendering - Single entity per chunk instead of individual block entities
2. Face culling - Only render visible block faces
3. Vertex-based geometry - Direct vertex buffer instead of Button entities
4. Occlusion caching - Cache solid checks during mesh generation
5. Reduced draw calls - Dramatically fewer entities in scene graph
"""

import numpy as np
from ursina import *
from ursina.shaders import lit_with_shadows_shader
from perlin_noise import PerlinNoise
import random
import math
from collections import deque
from ursina import color as ursina_color

# --- Configuration ---
CHUNK_SIZE = 16
WORLD_HEIGHT = 384
WORLD_BOTTOM = -64
RENDER_DISTANCE = 4  # Radius in chunks
BLOCK_SCALE = 1.0

# Block IDs
AIR = 0
STONE = 1
DIRT = 2
GRASS = 3
BEDROCK = 4
SAND = 5
WOOD = 6
LEAVES = 7
COAL_ORE = 8
IRON_ORE = 9
DIAMOND_ORE = 10
GOLD_ORE = 11
DEEPSLATE = 12
WATER = 13
CRAFTING_TABLE = 14

# Block Properties
BLOCK_NAMES = {
    0: 'Air', 1: 'Stone', 2: 'Dirt', 3: 'Grass', 4: 'Bedrock',
    5: 'Sand', 6: 'Oak Log', 7: 'Oak Leaves', 8: 'Coal Ore',
    9: 'Iron Ore', 10: 'Diamond Ore', 11: 'Gold Ore', 12: 'Deepslate',
    13: 'Water', 14: 'Crafting Table'
}

BLOCK_COLORS = {
    0: ursina_color.clear, 1: ursina_color.gray, 2: ursina_color.rgb(101, 67, 33), 3: ursina_color.rgb(79, 132, 46),
    4: ursina_color.black, 5: ursina_color.rgb(237, 220, 142), 6: ursina_color.rgb(84, 53, 26),
    7: ursina_color.rgb(46, 113, 36), 8: ursina_color.rgb(50, 50, 50), 9: ursina_color.rgb(217, 166, 127),
    10: ursina_color.rgb(100, 227, 206), 11: ursina_color.rgb(255, 217, 64), 12: ursina_color.rgb(30, 30, 40),
    13: ursina_color.rgba(41, 102, 237, 200), 14: ursina_color.rgb(109, 76, 46)
}

# Hardness (seconds to break with hand)
BLOCK_HARDNESS = {
    0: 0, 1: 1.5, 2: 0.5, 3: 0.6, 4: -1, 5: 0.5, 6: 2.0, 7: 0.2,
    8: 3.0, 9: 3.0, 10: 3.0, 11: 3.0, 12: 1.5, 13: -1, 14: 2.5
}

# Tool Multipliers (Hand = 1)
TOOL_MULTIPLIERS = {
    'pickaxe': {1: 1, 8: 4, 9: 4, 10: 4, 11: 4, 12: 4}, # Stone, Ores, Deepslate
    'shovel': {2: 4, 3: 4, 5: 4}, # Dirt, Grass, Sand
    'axe': {6: 4, 14: 4} # Wood, Crafting Table
}

# --- Noise Generation ---
noise_terrain = PerlinNoise(octaves=4, seed=1)
noise_caves = PerlinNoise(octaves=2, seed=2)
noise_temp = PerlinNoise(octaves=2, seed=3)
noise_humid = PerlinNoise(octaves=2, seed=4)

def get_biome(x, z):
    t = noise_temp([x * 0.002, z * 0.002])
    h = noise_humid([x * 0.002, z * 0.002])
    if t > 0.5 and h < -0.2: return 'desert'
    if t > 0.0 and h > 0.5: return 'forest'
    if t < -0.2 and h > 0.2: return 'taiga'
    if t < -0.5: return 'mountains'
    return 'plains'

def generate_chunk_data(cx, cz):
    """Generates the 3D array for a chunk"""
    data = np.zeros((CHUNK_SIZE, WORLD_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
    light = np.ones((CHUNK_SIZE, WORLD_HEIGHT, CHUNK_SIZE), dtype=np.uint8) * 15 # Sky light default
    
    for x in range(CHUNK_SIZE):
        for z in range(CHUNK_SIZE):
            wx, wz = cx * CHUNK_SIZE + x, cz * CHUNK_SIZE + z
            
            # Terrain Height
            n = noise_terrain([wx * 0.01, wz * 0.01])
            height = int(64 + n * 20)
            biome = get_biome(wx, wz)
            
            # Cave Noise
            cave_n = noise_caves([wx * 0.05, 0, wz * 0.05]) # Simplified 3D
            
            for y in range(WORLD_BOTTOM, WORLD_BOTTOM + WORLD_HEIGHT):
                idx = y - WORLD_BOTTOM
                if idx < 0 or idx >= WORLD_HEIGHT: continue
                
                block = AIR
                
                # Bedrock
                if y == WORLD_BOTTOM:
                    block = BEDROCK
                # Caves
                elif y < 60 and abs(cave_n) > 0.6:
                    block = AIR
                # Surface Logic
                elif y <= height:
                    if biome == 'desert':
                        if y == height: block = SAND
                        elif y > height - 4: block = SAND
                        else: block = STONE
                    else:
                        if y == height: block = GRASS
                        elif y > height - 4: block = DIRT
                        else: block = DEEPSLATE if y < 0 else STONE
                    
                    # Ores (Simple scatter)
                    if block in [STONE, DEEPSLATE] and y < 64:
                        if random.Random(wx * wz + y).random() < 0.02: block = COAL_ORE
                        if random.Random(wx * wz + y + 1).random() < 0.015: block = IRON_ORE
                        if y < -16 and random.Random(wx * wz + y + 2).random() < 0.005: block = DIAMOND_ORE
                
                # Water
                if block == AIR and y <= 63 and y > height:
                    block = WATER
                
                data[x, idx, z] = block

    return data

# --- Chunk Class ---
class Chunk(Entity):
    def __init__(self, cx, cz, world):
        super().__init__()
        self.cx, self.cz = cx, cz
        self.world = world
        self.data = None
        self.entity = None  # Single merged mesh entity
        self.is_generated = False
        
        # Position chunk in world space
        self.x = cx * CHUNK_SIZE
        self.z = cz * CHUNK_SIZE
        
    def generate(self):
        if self.is_generated: return
        self.data = generate_chunk_data(self.cx, self.cz)
        self.build_mesh()
        self.is_generated = True
        
    def build_mesh(self):
        # Destroy old mesh entity
        if self.entity:
            destroy(self.entity)
            self.entity = None
        
        # Generate optimized mesh data
        vertices = []
        colors = []
        block_data = []  # Store block info for raycasting
        
        # Optimization: Pre-calculate solid check lookup
        solid_cache = {}
        
        for x in range(CHUNK_SIZE):
            for z in range(CHUNK_SIZE):
                for y in range(WORLD_HEIGHT):
                    bid = self.data[x, y, z]
                    if bid == AIR:
                        continue
                    
                    # Check if surrounded (face culling)
                    if self._is_fully_occluded(x, y, z, solid_cache):
                        continue
                        
                    # Add vertex data for visible faces
                    wx, wy, wz = self.x + x, y + WORLD_BOTTOM, self.z + z
                    face_verts, face_colors = self._generate_block_faces(wx, wy, wz, bid)
                    vertices.extend(face_verts)
                    colors.extend(face_colors)
                    
                    # Store block position for interaction
                    block_data.append((wx, wy, wz, bid))
        
        # Create single merged entity
        if vertices:
            self.entity = Entity(
                parent=scene,
                model=Mesh(vertices=vertices, colors=colors, mode='triangle'),
                position=(0, 0, 0),
                collider='box'
            )
            self.entity.block_data = block_data
            self.entity.chunk = self
        
    def _is_fully_occluded(self, x, y, z, cache):
        """Check if block is completely surrounded by solid blocks"""
        key = (x, y, z)
        if key in cache:
            return cache[key]
        
        directions = [
            (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0),
            (0, 0, 1), (0, 0, -1)
        ]
        
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if not self._is_solid_local(nx, ny, nz):
                cache[key] = False
                return False
        
        cache[key] = True
        return True
    
    def _is_solid_local(self, x, y, z):
        """Check if a local coordinate is solid"""
        if x < 0 or x >= CHUNK_SIZE or z < 0 or z >= CHUNK_SIZE:
            return True  # Treat edge neighbors as solid for culling
        if y < 0 or y >= WORLD_HEIGHT:
            return False
        bid = self.data[x, y, z]
        return bid != AIR and bid != WATER
    
    def update_block(self, x, y, z, bid):
        """Update a single block and rebuild mesh"""
        if 0 <= x < CHUNK_SIZE and 0 <= z < CHUNK_SIZE and 0 <= y < WORLD_HEIGHT:
            self.data[x, y, z] = bid
            # Rebuild chunk mesh
            self.build_mesh()
    
    def _generate_block_faces(self, x, y, z, bid):
        """Generate vertices for visible faces of a block"""
        verts = []
        cols = []
        block_color = BLOCK_COLORS.get(bid, ursina_color.white)
        
        # Define face vertices (local coordinates)
        faces = [
            # Top
            [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
            # Bottom
            [(0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)],
            # North
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
            # South
            [(1, 0, 1), (0, 0, 1), (0, 1, 1), (1, 1, 1)],
            # East
            [(1, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)],
            # West
            [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
        ]
        
        # Check which faces are visible
        face_checks = [
            (0, 1, 0), (0, -1, 0), (0, 0, -1),
            (0, 0, 1), (1, 0, 0), (-1, 0, 0)
        ]
        
        for i, (dx, dy, dz) in enumerate(face_checks):
            if self._is_solid_local(x + dx, y + dy, z + dz):
                continue
            
            # Add face vertices
            for vx, vy, vz in faces[i]:
                verts.extend([x + vx, y + vy, z + vz])
                cols.append(color)
        
        return verts, cols

# --- Player Controller ---
class Player(Entity):
    def __init__(self, world=None, **kwargs):
        super().__init__(**kwargs)
        self.world = world
        self.cursor = Entity(parent=camera.ui, model='quad', scale=0.008, color=ursina_color.red)
        self.speed = 5
        self.jump_force = 8
        self.gravity = 20
        self.velocity = Vec3(0,0,0)
        self.grounded = False
        self.health = 20
        self.hunger = 20
        self.inventory = [3] * 9 # Hotbar
        self.selected_slot = 0
        self.camera_pivot = Entity(parent=self, y=1.62) # Eye height
        camera.parent = self.camera_pivot
        
        # Mining
        self.mining_target = None
        self.mining_progress = 0
        self.breaking_sound = Audio('assets/punch', autoplay=False, loop=True) if hasattr(Audio, '__init__') else None

    def input(self, key):
        if key == 'left mouse down' and not ui.visible:
            self.start_mining()
        if key == 'left mouse up':
            self.stop_mining()
        if key == 'right mouse down' and not ui.visible:
            self.place_block()
        if key == 'e':
            ui.visible = not ui.visible
            mouse.locked = not ui.visible
        if key in ['1','2','3','4','5','6','7','8','9']:
            self.selected_slot = int(key) - 1

    def start_mining(self):
        hit = raycast(camera.world_position, camera.forward, distance=5)
        if hit.hit:
            # Check if hitting a chunk mesh
            if hasattr(hit.entity, 'block_data'):
                # Find closest block from hit position
                block_data = hit.entity.block_data
                chunk = hit.entity.chunk
                self.mining_target = {
                    'entity': hit.entity,
                    'hit_pos': hit.world_point,
                    'chunk': chunk
                }
                self.mining_progress = 0

    def stop_mining(self):
        self.mining_target = None
        self.mining_progress = 0

    def place_block(self):
        hit = raycast(camera.world_position, camera.forward, distance=5)
        if hit.hit:
            if hasattr(hit.entity, 'block_data'):
                pos = hit.world_point + hit.normal * 0.5
                # Snap to grid
                pos = Vec3(round(pos.x), round(pos.y), round(pos.z))
            else:
                return
                
            # Don't place inside player
            if distance_xz(pos, self.position) < 0.8 and abs(pos.y - self.y) < 1.8:
                return
            
            bid = self.inventory[self.selected_slot]
            # Find chunk
            lx, ly, lz = int(pos.x % CHUNK_SIZE), int(pos.y - WORLD_BOTTOM), int(pos.z % CHUNK_SIZE)
            cx, cz = int(pos.x // CHUNK_SIZE), int(pos.z // CHUNK_SIZE)
            chunk = self.world.get_chunk(cx, cz)
            if chunk:
                chunk.update_block(lx, ly, lz, bid)

    def update(self):
        if ui.visible: return
        
        # Movement
        direction = Vec3(0,0,0)
        if held_keys['w']: direction += camera.forward
        if held_keys['s']: direction -= camera.forward
        if held_keys['d']: direction += camera.right
        if held_keys['a']: direction -= camera.right
        
        if direction != Vec3(0,0,0):
            direction = direction.normalized()
            
        # X/Z Movement with simple collision
        target_pos = self.position + direction * self.speed * time.dt
        if not self.check_collision(target_pos.x, self.y, self.z):
            self.x = target_pos.x
        if not self.check_collision(self.x, self.y, target_pos.z):
            self.z = target_pos.z

        # Gravity & Jump
        if held_keys['space'] and self.grounded:
            self.velocity.y = self.jump_force
            self.grounded = False
            
        self.velocity.y -= self.gravity * time.dt
        dy = self.velocity.y * time.dt
        
        if not self.check_collision(self.x, self.y + dy, self.z):
            self.y += dy
            self.grounded = False
        else:
            if self.velocity.y < 0: self.grounded = True
            self.velocity.y = 0
            # Snap to grid
            self.y = round(self.y)

        # Mining Logic
        if self.mining_target:
            # Find closest block to hit position
            chunk = self.mining_target['chunk']
            hit_pos = self.mining_target['hit_pos']
            
            # Calculate block coordinates from hit position
            bx, by, bz = int(hit_pos.x), int(hit_pos.y), int(hit_pos.z)
            lx = bx % CHUNK_SIZE
            lz = bz % CHUNK_SIZE
            ly = by - WORLD_BOTTOM
            
            if 0 <= lx < CHUNK_SIZE and 0 <= lz < CHUNK_SIZE and 0 <= ly < WORLD_HEIGHT:
                bid = chunk.data[lx, ly, lz]
                hardness = BLOCK_HARDNESS.get(bid, 1.0)
                if hardness > 0:
                    tool_mult = 1.0  # Simplified tool logic
                    self.mining_progress += time.dt * tool_mult / hardness
                    # Visual crack could go here
                    if self.mining_progress >= 1.0:
                        # Break block
                        chunk.update_block(lx, ly, lz, AIR)
                        self.stop_mining()
                else:
                    self.stop_mining()  # Unbreakable

    def check_collision(self, x, y, z):
        # Simple AABB vs Voxel check
        width = 0.3
        height = 1.8
        min_x, max_x = x - width, x + width
        min_y, max_y = y, y + height
        min_z, max_z = z - width, z + width
        
        # Check surrounding chunks/blocks
        # (Simplified: checks integer coordinates)
        for bx in range(int(min_x), int(max_x) + 1):
            for by in range(int(min_y), int(max_y) + 1):
                for bz in range(int(min_z), int(max_z) + 1):
                    cx, cz = int(bx // CHUNK_SIZE), int(bz // CHUNK_SIZE)
                    chunk = self.world.get_chunk(cx, cz)
                    if chunk:
                        lx, ly, lz = int(bx % CHUNK_SIZE), int(by - WORLD_BOTTOM), int(bz % CHUNK_SIZE)
                        if 0 <= lx < CHUNK_SIZE and 0 <= lz < CHUNK_SIZE and 0 <= ly < WORLD_HEIGHT:
                            bid = chunk.data[lx, ly, lz]
                            if bid != AIR and bid != WATER:
                                return True
        return False

# --- World Manager ---
class World(Entity):
    def __init__(self):
        super().__init__()
        self.chunks = {}
        
    def get_chunk(self, cx, cz):
        return self.chunks.get((cx, cz))
        
    def update(self):
        px, pz = int(player.x // CHUNK_SIZE), int(player.z // CHUNK_SIZE)
        
        # Load chunks
        for x in range(px - RENDER_DISTANCE, px + RENDER_DISTANCE + 1):
            for z in range(pz - RENDER_DISTANCE, pz + RENDER_DISTANCE + 1):
                if (x, z) not in self.chunks:
                    chunk = Chunk(x, z, self)
                    chunk.generate()
                    self.chunks[(x, z)] = chunk
        
        # Unload far chunks
        to_remove = []
        for (cx, cz), chunk in self.chunks.items():
            dist = math.sqrt((cx-px)**2 + (cz-pz)**2)
            if dist > RENDER_DISTANCE + 2:
                destroy(chunk)
                to_remove.append((cx, cz))
        for k in to_remove:
            del self.chunks[k]

# --- UI ---
class InventoryUI(Entity):
    def __init__(self):
        super().__init__(parent=camera.ui, visible=False)
        self.bg = Entity(parent=self, model='quad', scale=(0.8, 0.6), color=ursina_color.rgba(0,0,0,150))
        self.slots = []
        for i in range(9):
            slot = Entity(parent=self, model='quad', scale=0.05, position=(-0.35 + i*0.08, -0.25), color=ursina_color.gray)
            self.slots.append(slot)
            
    def update(self):
        for i, slot in enumerate(self.slots):
            # Highlight selected
            if i == player.selected_slot:
                slot.color = ursina_color.white
            else:
                slot.color = ursina_color.gray

# --- Setup ---
app = Ursina()
window.title = 'Python Minecraft Clone'
window.borderless = False
window.fullscreen = False
window.exit_button.visible = False

Sky(texture='sky_default')
world = World()
player = Player(world=world, position=(0, 100, 0))
ui = InventoryUI()

# Initial generation spawn
print("Generating initial chunks...")
for x in range(-2, 3):
    for z in range(-2, 3):
        c = Chunk(x, z, world)
        c.generate()
        world.chunks[(x,z)] = c

player.y = 100 # Drop player from sky

app.run()
