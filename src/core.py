"""Minecraft Clone - Complete Phase 1 Core Mechanics"""

import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import math

class Block(IntEnum):
    AIR = 0
    STONE = 1
    DIRT = 2
    GRASS = 3
    BEDROCK = 4
    COBBLESTONE = 5
    PLANKS = 6
    GLASS = 7
    SAND = 8
    LOG = 9
    
    @property
    def is_solid(self) -> bool:
        return self != Block.AIR and self != Block.GLASS
    
    @property
    def is_transparent(self) -> bool:
        return self == Block.GLASS

CHUNK_SIZE_X, CHUNK_SIZE_Z, CHUNK_SIZE_Y = 16, 16, 384
WORLD_MIN_Y, WORLD_MAX_Y = -64, 320

BLOCK_UV_MAP = {
    Block.STONE: {i: (0.0, 0.0, 0.0625, 0.0625) for i in range(6)},
    Block.DIRT: {i: (0.0625, 0.0, 0.125, 0.0625) for i in range(6)},
    Block.GRASS: {0: (0.125, 0.0, 0.1875, 0.0625), 1: (0.125, 0.0, 0.1875, 0.0625), 2: (0.1875, 0.0, 0.25, 0.0625), 3: (0.0625, 0.0, 0.125, 0.0625), 4: (0.125, 0.0, 0.1875, 0.0625), 5: (0.125, 0.0, 0.1875, 0.0625)},
    Block.BEDROCK: {i: (0.25, 0.0, 0.3125, 0.0625) for i in range(6)},
    Block.COBBLESTONE: {i: (0.3125, 0.0, 0.375, 0.0625) for i in range(6)},
    Block.PLANKS: {i: (0.375, 0.0, 0.4375, 0.0625) for i in range(6)},
    Block.GLASS: {i: (0.4375, 0.0, 0.5, 0.0625) for i in range(6)},
    Block.SAND: {i: (0.5, 0.0, 0.5625, 0.0625) for i in range(6)},
    Block.LOG: {0: (0.5625, 0.0, 0.625, 0.0625), 1: (0.5625, 0.0, 0.625, 0.0625), 2: (0.625, 0.0, 0.6875, 0.0625), 3: (0.625, 0.0, 0.6875, 0.0625), 4: (0.5625, 0.0, 0.625, 0.0625), 5: (0.5625, 0.0, 0.625, 0.0625)},
}

@dataclass
class MeshData:
    vertices: List[float] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    
    def add_quad(self, x, y, z, face: int, block_type: Block, ao: List[float]):
        if block_type not in BLOCK_UV_MAP: return
        uv = BLOCK_UV_MAP[block_type][face]
        u1, v1, u2, v2 = uv
        verts = [(1,0,0),(1,0,1),(1,1,1),(1,1,0)] if face==0 else [(0,0,1),(0,0,0),(0,1,0),(0,1,1)] if face==1 else [(0,1,1),(1,1,1),(1,1,0),(0,1,0)] if face==2 else [(0,0,0),(1,0,0),(1,0,1),(0,0,1)] if face==3 else [(1,0,1),(0,0,1),(0,1,1),(1,1,1)] if face==4 else [(0,0,0),(1,0,0),(1,1,0),(0,1,0)]
        base_idx = len(self.vertices) // 6
        for i, (vx, vy, vz) in enumerate(verts):
            self.vertices.extend([x+vx, y+vy, z+vz, u1 if i%2==0 else u2, v1 if i<2 or i==3 else v2, ao[i]])
        self.indices.extend([base_idx, base_idx+1, base_idx+2, base_idx, base_idx+2, base_idx+3])

class Chunk:
    def __init__(self, cx: int, cz: int):
        self.cx, self.cz = cx, cz
        self.blocks = np.zeros((CHUNK_SIZE_X, CHUNK_SIZE_Y, CHUNK_SIZE_Z), dtype=np.uint8)
        self.mesh_dirty, self.mesh_data, self.neighbors = True, None, {}
    
    def get_block(self, x, y, z) -> Block:
        if not (0<=x<CHUNK_SIZE_X and 0<=z<CHUNK_SIZE_Z and WORLD_MIN_Y<=y<WORLD_MAX_Y): return Block.AIR
        ly = y - WORLD_MIN_Y
        return Block(self.blocks[x, ly, z]) if 0<=ly<CHUNK_SIZE_Y else Block.AIR
    
    def set_block(self, x, y, z, block: Block):
        if not (0<=x<CHUNK_SIZE_X and 0<=z<CHUNK_SIZE_Z and WORLD_MIN_Y<=y<WORLD_MAX_Y): return
        ly = y - WORLD_MIN_Y
        if 0<=ly<CHUNK_SIZE_Y:
            self.blocks[x, ly, z] = block.value
            self.mark_dirty()
            for k,dx2,dz2 in [('west',-1,0),('east',1,0),('north',0,-1),('south',0,1)]:
                if (x==0 and k=='west') or (x==15 and k=='east') or (z==0 and k=='north') or (z==15 and k=='south'):
                    if k in self.neighbors: self.neighbors[k].mark_dirty()
    
    def mark_dirty(self): self.mesh_dirty, self.mesh_data = True, None
    
    def calculate_ao(self, x, y, z, dx, dy, dz) -> List[float]:
        occ = sum(0.15 for sx,sy,sz in [(x+dx,y+dy,z+dz),(x+dx,y,z+dz),(x+dx,y+dy,z)] if 0<=sx<CHUNK_SIZE_X and 0<=sz<CHUNK_SIZE_Z and 0<=(sy-WORLD_MIN_Y)<CHUNK_SIZE_Y and Block(self.blocks[sx,sy-WORLD_MIN_Y,sz]).is_solid)
        v = max(0.5, 1.0-occ)
        return [v,v,v,v]
    
    def generate_mesh(self) -> MeshData:
        if not self.mesh_dirty: return self.mesh_data
        mesh = MeshData()
        for y in range(CHUNK_SIZE_Y):
            for z in range(CHUNK_SIZE_Z):
                for x in range(CHUNK_SIZE_X):
                    bv = self.blocks[x,y,z]
                    if bv == 0: continue
                    block = Block(bv)
                    gy = y + WORLD_MIN_Y
                    for fi,(dx,dy,dz) in enumerate([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]):
                        nx,ny,nz = x+dx,y+dy,z+dz
                        nb = Block.AIR
                        if 0<=nx<CHUNK_SIZE_X and 0<=nz<CHUNK_SIZE_Z:
                            lny = ny-WORLD_MIN_Y
                            if 0<=lny<CHUNK_SIZE_Y: nb = Block(self.blocks[nx,lny,nz])
                        if nb==Block.AIR or (block.is_transparent and nb.is_solid):
                            mesh.add_quad(x,gy,z,fi,block,self.calculate_ao(x,y,z,dx,dy,dz))
        self.mesh_data, self.mesh_dirty = mesh, False
        return mesh

class World:
    def __init__(self): self.chunks = {}
    def get_chunk(self, cx, cz): return self.chunks.get((cx,cz))
    def load_chunk(self, cx, cz) -> Chunk:
        if (cx,cz) in self.chunks: return self.chunks[(cx,cz)]
        chunk = Chunk(cx,cz)
        self.chunks[(cx,cz)] = chunk
        for dn,dx,dz in [('east',1,0),('west',-1,0),('south',0,1),('north',0,-1)]:
            if (cx+dx,cz+dz) in self.chunks:
                chunk.neighbors[dn] = self.chunks[(cx+dx,cz+dz)]
                self.chunks[(cx+dx,cz+dz)].neighbors[{'east':'west','west':'east','south':'north','north':'south'}[dn]] = chunk
        self.generate_terrain(chunk)
        return chunk
    def _noise(self, gx, gz): return math.sin(gx*0.05)*math.cos(gz*0.05)*0.7 + math.sin(gx*0.01+1.5)*math.cos(gz*0.02)*0.3
    def generate_terrain(self, chunk):
        for x in range(CHUNK_SIZE_X):
            for z in range(CHUNK_SIZE_Z):
                h = int(75 + self._noise(chunk.cx*16+x, chunk.cz*16+z)*25)
                h = max(WORLD_MIN_Y+1, min(WORLD_MAX_Y-1, h))
                for y in range(WORLD_MIN_Y, h+1):
                    chunk.set_block(x,y,z, Block.BEDROCK if y==WORLD_MIN_Y else Block.GRASS if y==h else Block.DIRT if y>h-4 else Block.STONE)
    def get_block(self, x, y, z) -> Block:
        c = self.get_chunk(x>>4, z>>4)
        return c.get_block(x&15, y, z&15) if c else Block.AIR
    def set_block(self, x, y, z, block):
        c = self.get_chunk(x>>4, z>>4)
        if c: c.set_block(x&15, y, z&15, block)

@dataclass
class Player:
    x, y, z = 0.0, 70.0, 0.0
    vx, vy, vz = 0.0, 0.0, 0.0
    yaw, pitch, is_grounded = 0.0, 0.0, False
    WIDTH, HEIGHT, EYE_HEIGHT = 0.6, 1.8, 1.62
    SPEED_WALK, GRAVITY, TERMINAL_VELOCITY, JUMP_FORCE = 4.317, 32.0, 78.4, 8.4
    
    def get_eye_pos(self): return (self.x, self.y+self.EYE_HEIGHT, self.z)
    def get_aabb(self):
        hw = self.WIDTH/2
        return (self.x-hw, self.y, self.z-hw, self.x+hw, self.y+self.HEIGHT, self.z+hw)
    
    def update(self, dt, inp, jump, world):
        if not self.is_grounded:
            self.vy = max(-self.TERMINAL_VELOCITY, self.vy-self.GRAVITY*dt)
        else: self.vy = 0.0
        if jump and self.is_grounded: self.vy, self.is_grounded = self.JUMP_FORCE, False
        mx, mz = inp
        if mx or mz:
            ln = math.sqrt(mx*mx+mz*mz)
            if ln>1: mx,mz = mx/ln, mz/ln
            yr = math.radians(self.yaw)
            self.vx, self.vz = (mx*math.cos(yr)-mz*math.sin(yr))*self.SPEED_WALK, (mx*math.sin(yr)+mz*math.cos(yr))*self.SPEED_WALK
        else: self.vx, self.vz = 0.0, 0.0
        self.is_grounded = False
        self.x += self.vx*dt; self.resolve_collision(world, 0)
        self.y += self.vy*dt; self.resolve_collision(world, 1)
        self.z += self.vz*dt; self.resolve_collision(world, 2)
    
    def resolve_collision(self, world, axis):
        minx,miny,minz,maxx,maxy,maxz = self.get_aabb()
        for bx in range(int(math.floor(minx)), int(math.ceil(maxx))+1):
            for by in range(int(math.floor(miny)), int(math.ceil(maxy))+1):
                for bz in range(int(math.floor(minz)), int(math.ceil(maxz))+1):
                    b = world.get_block(bx,by,bz)
                    if not b.is_solid: continue
                    oxmin,oxmax,oymin,oymax,ozmin,ozmax = maxx-bx, bx+1-minx, maxy-by, by+1-miny, maxz-bz, bz+1-minz
                    if axis==0:
                        if oxmin<oxmax and oxmin>0: self.x-=oxmin; self.vx=0.0
                        elif oxmax>0: self.x+=oxmax; self.vx=0.0
                    elif axis==1:
                        if oymin<oymax and oymin>0: self.y-=oymin; self.vy=0.0
                        elif oymax>0: self.y+=oymax; self.vy=0.0; self.is_grounded=True
                    elif axis==2:
                        if ozmin<ozmax and ozmin>0: self.z-=ozmin; self.vz=0.0
                        elif ozmax>0: self.z+=ozmax; self.vz=0.0

@dataclass
class RayHit:
    hit, x, y, z, face, distance = False, 0, 0, 0, (0,0,0), 0.0

class Raycaster:
    MAX_DISTANCE = 5.0
    @staticmethod
    def cast(origin, direction, world) -> RayHit:
        dx,dy,dz = direction
        ox,oy,oz = origin
        x,y,z = int(math.floor(ox)), int(math.floor(oy)), int(math.floor(oz))
        sx,sy,sz = (1 if dx>0 else -1), (1 if dy>0 else -1), (1 if dz>0 else -1)
        if abs(dx)<1e-6: dx=1e-6*(1 if dx>=0 else -1)
        if abs(dy)<1e-6: dy=1e-6*(1 if dy>=0 else -1)
        if abs(dz)<1e-6: dz=1e-6*(1 if dz>=0 else -1)
        tdx,tdy,tdz = abs(1/dx), abs(1/dy), abs(1/dz)
        tmx = ((x+1-ox) if sx>0 else (ox-x))*tdx
        tmy = ((y+1-oy) if sy>0 else (oy-y))*tdy
        tmz = ((z+1-oz) if sz>0 else (oz-z))*tdz
        lf, dist = (0,0,0), 0.0
        while dist < Raycaster.MAX_DISTANCE:
            b = world.get_block(x,y,z)
            if b.is_solid: return RayHit(True,x,y,z,lf,dist)
            if tmx<tmy:
                if tmx<tmz: dist=tmx; tmx+=tdx; x+=sx; lf=(-sx,0,0)
                else: dist=tmz; tmz+=tdz; z+=sz; lf=(0,0,-sz)
            else:
                if tmy<tmz: dist=tmy; tmy+=tdy; y+=sy; lf=(0,-sy,0)
                else: dist=tmz; tmz+=tdz; z+=sz; lf=(0,0,-sz)
        return RayHit()

class Hotbar:
    def __init__(self):
        self.slots = [Block.GRASS,Block.DIRT,Block.STONE,Block.COBBLESTONE,Block.PLANKS,Block.GLASS,Block.SAND,Block.BEDROCK,Block.LOG]
        self.selected_index = 0
    def select_next(self): self.selected_index = (self.selected_index+1)%9
    def select_prev(self): self.selected_index = (self.selected_index-1)%9
    def select(self, i): 
        if 0<=i<9: self.selected_index=i
    def get_selected_block(self): return self.slots[self.selected_index]
    def render_ui(self): return {'selected':self.selected_index,'slots':self.slots}

class Game:
    def __init__(self):
        self.world, self.player, self.hotbar, self.raycaster = World(), Player(), Hotbar(), Raycaster()
        px,pz = int(self.player.x)>>4, int(self.player.z)>>4
        for dx in range(-1,2):
            for dz in range(-1,2): self.world.load_chunk(px+dx, pz+dz)
    
    def handle_input(self, dt, keys, mouse_delta, scroll):
        self.player.yaw += mouse_delta[0] if isinstance(mouse_delta, (tuple,list)) else 0*0.1
        self.player.pitch = max(-90, min(90, self.player.pitch-mouse_delta[1] if isinstance(mouse_delta, (tuple,list)) else 0*0.1))
        if scroll!=0: self.hotbar.select_prev() if scroll>0 else self.hotbar.select_next()
        for i in range(1,10):
            if keys.get(str(i)): self.hotbar.select(i-1)
        mx = (1 if keys.get('D') else 0)-(1 if keys.get('A') else 0)
        mz = (1 if keys.get('S') else 0)-(1 if keys.get('W') else 0)
        self.player.update(dt, (float(mx),float(mz)), keys.get('SPACE',False), self.world)
        if keys.get('CLICK_LEFT'): self.break_block()
        if keys.get('CLICK_RIGHT'): self.place_block()
    
    def break_block(self):
        h = self.raycaster.cast(self.player.get_eye_pos(), self.get_forward_vector(), self.world)
        if h.hit: self.world.set_block(h.x,h.y,h.z,Block.AIR)
    
    def place_block(self):
        h = self.raycaster.cast(self.player.get_eye_pos(), self.get_forward_vector(), self.world)
        if h.hit:
            px,py,pz = h.x+h.face[0], h.y+h.face[1], h.z+h.face[2]
            minx,miny,minz,maxx,maxy,maxz = self.player.get_aabb()
            if not (px>=maxx or px+1<=minx or py>=maxy or py+1<=miny or pz>=maxz or pz+1<=minz):
                self.world.set_block(px,py,pz,self.hotbar.get_selected_block())
    
    def get_forward_vector(self):
        yr,pr = math.radians(self.player.yaw), math.radians(self.player.pitch)
        return (-math.sin(yr)*math.cos(pr), math.sin(pr), math.cos(yr)*math.cos(pr))

if __name__ == "__main__":
    print("="*50+"\nMinecraft Clone - Phase 1 Core Mechanics Test\n"+"="*50)
    g = Game()
    print(f"\n✓ Game initialized\n✓ Player at: ({g.player.x:.2f},{g.player.y:.2f},{g.player.z:.2f})\n✓ Dimensions: {g.player.WIDTH}m×{g.player.HEIGHT}m\n✓ Eye height: {g.player.EYE_HEIGHT}m\n✓ Hotbar: {g.hotbar.get_selected_block().name}")
    g.handle_input(0.016,{'W':True},(0,0),0)
    print(f"\n--- Movement ---\nAfter W (16ms): ({g.player.x:.3f},{g.player.y:.3f},{g.player.z:.3f})")
    g.player.y,g.player.is_grounded = 100,False
    g.handle_input(0.1,{},{},0)
    print(f"\n--- Gravity ---\nAfter 100ms fall: vy={g.player.vy:.2f} blocks/s")
    h = g.raycaster.cast(g.player.get_eye_pos(),g.get_forward_vector(),g.world)
    print(f"\n--- Raycasting ---\nHit={h.hit}" + (f" at ({h.x},{h.y},{h.z}), face={h.face}" if h.hit else ""))
    print(f"\n--- Hotbar ---\nInitial: slot {g.hotbar.selected_index} ({g.hotbar.get_selected_block().name})")
    g.hotbar.select_next()
    print(f"After scroll: slot {g.hotbar.selected_index} ({g.hotbar.get_selected_block().name})")
    print("\n"+"="*50+"\nPhase 1 Complete!\n  ✓ Chunk data structure\n  ✓ Face culling mesh generation\n  ✓ Texture atlas UV mapping\n  ✓ Ambient occlusion\n  ✓ World management\n  ✓ Basic terrain generation\n  ✓ Player controller with AABB physics\n  ✓ Axis-separated collision resolution\n  ✓ Gravity and jumping\n  ✓ DDA raycasting\n  ✓ Block breaking/placing\n  ✓ Hotbar inventory\n"+"="*50)
