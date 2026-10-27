"""
FLIP (Fluid-Implicit-Particle) 流体解算器 - 2D
=============================================
对比你的 SPH 解算器：
- SPH: 纯拉格朗日，粒子间直接算力，压力通过状态方程近似
- FLIP: 混合方法，粒子携带速度（拉格朗日），网格上做压力投影（欧拉）

核心流程：
1. P2G: 粒子速度插值到 MAC 网格
2. 外力: 网格速度 += gravity * dt
3. 压力求解: 泊松方程使速度场无散度
4. G2P: FLIP 方式更新粒子速度（只传增量）
5. 粒子位移

作者参考: Bridson "Fluid Simulation for Computer Graphics"
"""

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

ti.init(arch=ti.gpu, default_fp=ti.f32)

# ==================== 模拟参数 ====================
# 网格参数
GRID_RES = 64            # 网格分辨率 (64x64)
DX = 1.0 / GRID_RES     # 网格间距
INV_DX = GRID_RES        # 1/dx

# 粒子参数
MAX_PARTICLES = 20000
N_INIT = 0               # 初始粒子数（动态填充）

# 物理参数
GRAVITY = -9.8
DT = 0.02                # 时间步长
RHO = 1000.0             # 流体密度
FLIP_RATIO = 0.95        # FLIP/PIC 混合比 (1.0=纯FLIP, 0.0=纯PIC)

# 边界
DOMAIN_X = 1.0
DOMAIN_Y = 1.0

# ==================== 数据结构 ====================
# --- MAC 网格 (Marker-And-Cell) ---
# 速度分量存储在面心：
#   u (水平速度) 存在竖直面上: (GRID_RES+1) x GRID_RES
#   v (垂直速度) 存在水平面上: GRID_RES x (GRID_RES+1)
#   压力/标记存在格心: GRID_RES x GRID_RES

# 当前帧网格速度
u = ti.field(dtype=ti.f32, shape=(GRID_RES + 1, GRID_RES))      # 水平速度 (面心)
v = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES + 1))      # 垂直速度 (面心)

# 保存旧速度 (FLIP 需要计算增量: v_new - v_old)
u_old = ti.field(dtype=ti.f32, shape=(GRID_RES + 1, GRID_RES))
v_old = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES + 1))

# 权重累加 (P2G 时用于加权平均)
u_weight = ti.field(dtype=ti.f32, shape=(GRID_RES + 1, GRID_RES))
v_weight = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES + 1))

# 压力场 & 散度 & 单元格标记
pressure = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES))
divergence = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES))
cell_type = ti.field(dtype=ti.i32, shape=(GRID_RES, GRID_RES))
# cell_type: 0=AIR, 1=FLUID, 2=SOLID

CELL_AIR = 0
CELL_FLUID = 1
CELL_SOLID = 2

# --- 粒子 ---
particle_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PARTICLES)
particle_vel = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PARTICLES)
num_particles = ti.field(dtype=ti.i32, shape=())


# ==================== 工具函数 ====================
@ti.func
def clamp(x, low, high):
    return ti.max(low, ti.min(x, high))


@ti.func
def bilinear_interp_u(pos: ti.math.vec2) -> ti.f32:
    """
    在 MAC 网格的 u 分量位置做双线性插值
    u 存储在 (i, j+0.5) 位置，即竖直面的中心
    """
    # u 的采样坐标：x 方向从 0 开始，y 方向偏移 0.5 个格子
    px = pos.x * INV_DX
    py = pos.y * INV_DX - 0.5

    i = int(ti.floor(px))
    j = int(ti.floor(py))

    fx = px - i
    fy = py - j

    i = clamp(i, 0, GRID_RES - 1)
    j = clamp(j, 0, GRID_RES - 2)

    # 确保不越界
    i1 = ti.min(i + 1, GRID_RES)
    j1 = ti.min(j + 1, GRID_RES - 1)

    return (u[i, j] * (1 - fx) * (1 - fy) +
            u[i1, j] * fx * (1 - fy) +
            u[i, j1] * (1 - fx) * fy +
            u[i1, j1] * fx * fy)


@ti.func
def bilinear_interp_v(pos: ti.math.vec2) -> ti.f32:
    """
    在 MAC 网格的 v 分量位置做双线性插值
    v 存储在 (i+0.5, j) 位置，即水平面的中心
    """
    px = pos.x * INV_DX - 0.5
    py = pos.y * INV_DX

    i = int(ti.floor(px))
    j = int(ti.floor(py))

    fx = px - i
    fy = py - j

    i = clamp(i, 0, GRID_RES - 2)
    j = clamp(j, 0, GRID_RES - 1)

    i1 = ti.min(i + 1, GRID_RES - 1)
    j1 = ti.min(j + 1, GRID_RES)

    return (v[i, j] * (1 - fx) * (1 - fy) +
            v[i1, j] * fx * (1 - fy) +
            v[i, j1] * (1 - fx) * fy +
            v[i1, j1] * fx * fy)


@ti.func
def bilinear_interp_u_old(pos: ti.math.vec2) -> ti.f32:
    """插值旧的 u 速度场"""
    px = pos.x * INV_DX
    py = pos.y * INV_DX - 0.5

    i = int(ti.floor(px))
    j = int(ti.floor(py))

    fx = px - i
    fy = py - j

    i = clamp(i, 0, GRID_RES - 1)
    j = clamp(j, 0, GRID_RES - 2)

    i1 = ti.min(i + 1, GRID_RES)
    j1 = ti.min(j + 1, GRID_RES - 1)

    return (u_old[i, j] * (1 - fx) * (1 - fy) +
            u_old[i1, j] * fx * (1 - fy) +
            u_old[i, j1] * (1 - fx) * fy +
            u_old[i1, j1] * fx * fy)


@ti.func
def bilinear_interp_v_old(pos: ti.math.vec2) -> ti.f32:
    """插值旧的 v 速度场"""
    px = pos.x * INV_DX - 0.5
    py = pos.y * INV_DX

    i = int(ti.floor(px))
    j = int(ti.floor(py))

    fx = px - i
    fy = py - j

    i = clamp(i, 0, GRID_RES - 2)
    j = clamp(j, 0, GRID_RES - 1)

    i1 = ti.min(i + 1, GRID_RES - 1)
    j1 = ti.min(j + 1, GRID_RES)

    return (v_old[i, j] * (1 - fx) * (1 - fy) +
            v_old[i1, j] * fx * (1 - fy) +
            v_old[i, j1] * (1 - fx) * fy +
            v_old[i1, j1] * fx * fy)


# ==================== 步骤 1: 标记单元格类型 ====================
@ti.kernel
def mark_cells():
    """
    根据粒子位置标记网格单元类型:
    - SOLID: 边界墙壁
    - FLUID: 含有粒子的格子
    - AIR: 空的格子

    对比 SPH: SPH 不需要这步，因为它没有背景网格
    """
    # 先全部标记为 AIR
    for i, j in cell_type:
        if i == 0 or i == GRID_RES - 1 or j == 0 or j == GRID_RES - 1:
            cell_type[i, j] = CELL_SOLID
        else:
            cell_type[i, j] = CELL_AIR

    # 粒子所在格子标记为 FLUID
    for p in range(num_particles[None]):
        pos = particle_pos[p]
        i = int(pos.x * INV_DX)
        j = int(pos.y * INV_DX)
        i = clamp(i, 1, GRID_RES - 2)
        j = clamp(j, 1, GRID_RES - 2)
        cell_type[i, j] = CELL_FLUID


# ==================== 步骤 2: P2G (Particle to Grid) ====================
@ti.kernel
def particle_to_grid():
    """
    将粒子速度散布(splat)到 MAC 网格上
    使用双线性权重进行加权平均

    对比 SPH: 这相当于 SPH 的密度计算，但目的不同
    - SPH 算密度是为了求压力（状态方程）
    - FLIP 做 P2G 是为了在网格上做压力投影
    """
    # 清零
    for i, j in u:
        u[i, j] = 0.0
        u_weight[i, j] = 0.0
    for i, j in v:
        v[i, j] = 0.0
        v_weight[i, j] = 0.0

    # 散布粒子速度到网格
    for p in range(num_particles[None]):
        pos = particle_pos[p]
        vel = particle_vel[p]

        # --- 散布 u 分量 ---
        # u 采样点在 (i, j+0.5)
        px = pos.x * INV_DX
        py = pos.y * INV_DX - 0.5

        i = int(ti.floor(px))
        j = int(ti.floor(py))
        fx = px - i
        fy = py - j

        # 双线性权重 splat
        if 0 <= i < GRID_RES + 1 and 0 <= j < GRID_RES:
            w = (1 - fx) * (1 - fy)
            ti.atomic_add(u[i, j], w * vel.x)
            ti.atomic_add(u_weight[i, j], w)
        if 0 <= i + 1 < GRID_RES + 1 and 0 <= j < GRID_RES:
            w = fx * (1 - fy)
            ti.atomic_add(u[i + 1, j], w * vel.x)
            ti.atomic_add(u_weight[i + 1, j], w)
        if 0 <= i < GRID_RES + 1 and 0 <= j + 1 < GRID_RES:
            w = (1 - fx) * fy
            ti.atomic_add(u[i, j + 1], w * vel.x)
            ti.atomic_add(u_weight[i, j + 1], w)
        if 0 <= i + 1 < GRID_RES + 1 and 0 <= j + 1 < GRID_RES:
            w = fx * fy
            ti.atomic_add(u[i + 1, j + 1], w * vel.x)
            ti.atomic_add(u_weight[i + 1, j + 1], w)

        # --- 散布 v 分量 ---
        # v 采样点在 (i+0.5, j)
        px = pos.x * INV_DX - 0.5
        py = pos.y * INV_DX

        i = int(ti.floor(px))
        j = int(ti.floor(py))
        fx = px - i
        fy = py - j

        if 0 <= i < GRID_RES and 0 <= j < GRID_RES + 1:
            w = (1 - fx) * (1 - fy)
            ti.atomic_add(v[i, j], w * vel.y)
            ti.atomic_add(v_weight[i, j], w)
        if 0 <= i + 1 < GRID_RES and 0 <= j < GRID_RES + 1:
            w = fx * (1 - fy)
            ti.atomic_add(v[i + 1, j], w * vel.y)
            ti.atomic_add(v_weight[i + 1, j], w)
        if 0 <= i < GRID_RES and 0 <= j + 1 < GRID_RES + 1:
            w = (1 - fx) * fy
            ti.atomic_add(v[i, j + 1], w * vel.y)
            ti.atomic_add(v_weight[i, j + 1], w)
        if 0 <= i + 1 < GRID_RES and 0 <= j + 1 < GRID_RES + 1:
            w = fx * fy
            ti.atomic_add(v[i + 1, j + 1], w * vel.y)
            ti.atomic_add(v_weight[i + 1, j + 1], w)


@ti.kernel
def normalize_grid_velocity():
    """归一化网格速度（加权平均的除法步骤）"""
    for i, j in u:
        if u_weight[i, j] > 1e-6:
            u[i, j] /= u_weight[i, j]
        else:
            u[i, j] = 0.0

    for i, j in v:
        if v_weight[i, j] > 1e-6:
            v[i, j] /= v_weight[i, j]
        else:
            v[i, j] = 0.0


# ==================== 步骤 3: 保存旧速度 (FLIP 需要) ====================
@ti.kernel
def save_velocity():
    """
    保存压力投影前的速度
    FLIP 的关键: v_particle += (v_grid_new - v_grid_old)
    需要 v_grid_old 来算增量
    """
    for i, j in u:
        u_old[i, j] = u[i, j]
    for i, j in v:
        v_old[i, j] = v[i, j]


# ==================== 步骤 4: 施加外力 ====================
@ti.kernel
def apply_gravity():
    """
    在网格上施加重力
    只修改 v (垂直方向速度)

    对比 SPH: SPH 在 compute_forces 中直接加重力到粒子
    """
    for i, j in v:
        v[i, j] += GRAVITY * DT


# ==================== 步骤 5: 边界条件 ====================
@ti.kernel
def apply_boundary_conditions():
    """
    固体边界：法向速度置零

    对比 SPH: SPH 用"软边界"（弹簧力），FLIP 用硬边界（直接置零）
    """
    # 左右墙壁：u=0
    for j in range(GRID_RES):
        u[0, j] = 0.0
        u[1, j] = 0.0  # 第一层固体内部
        u[GRID_RES - 1, j] = 0.0
        u[GRID_RES, j] = 0.0

    # 上下墙壁：v=0
    for i in range(GRID_RES):
        v[i, 0] = 0.0
        v[i, 1] = 0.0
        v[i, GRID_RES - 1] = 0.0
        v[i, GRID_RES] = 0.0


# ==================== 步骤 6: 压力求解 ====================
@ti.kernel
def compute_divergence_field():
    """
    计算速度场的散度: div(u) = du/dx + dv/dy

    这是压力泊松方程的右端项:
    ∇²p = (ρ / dt) * ∇·u

    对比 SPH: SPH 完全没有这步！
    SPH 用状态方程 p = k(ρ - ρ0) 近似压力
    FLIP 精确求解泊松方程确保不可压缩性
    """
    for i, j in divergence:
        if cell_type[i, j] == CELL_FLUID:
            # MAC 网格上的散度计算非常简洁
            # div = (u_right - u_left)/dx + (v_top - v_bottom)/dy
            divergence[i, j] = -(u[i + 1, j] - u[i, j] +
                                  v[i, j + 1] - v[i, j]) * INV_DX
        else:
            divergence[i, j] = 0.0


@ti.kernel
def pressure_jacobi_iteration() -> ti.f32:
    """
    Jacobi 迭代求解压力泊松方程:
    ∇²p = rhs  =>  p(i,j) = (1/4) * [p(i+1,j) + p(i-1,j) + p(i,j+1) + p(i,j-1) - dx²*rhs]

    这是 FLIP 最耗时的步骤（Niagara 中的 Solve Pressure）
    通常需要 50-200 次迭代收敛

    对比 SPH: SPH 无需求解线性系统，这是 FLIP 的主要计算开销
    但换来的是精确的不可压缩性
    """
    residual = 0.0

    for i, j in pressure:
        if cell_type[i, j] == CELL_FLUID:
            # 统计有效邻居数 (非固体)
            n_neighbors = 0.0
            p_sum = 0.0

            # 左
            if i > 0:
                if cell_type[i - 1, j] != CELL_SOLID:
                    n_neighbors += 1.0
                    if cell_type[i - 1, j] == CELL_FLUID:
                        p_sum += pressure[i - 1, j]
                    # AIR: pressure = 0 (Dirichlet)
            # 右
            if i < GRID_RES - 1:
                if cell_type[i + 1, j] != CELL_SOLID:
                    n_neighbors += 1.0
                    if cell_type[i + 1, j] == CELL_FLUID:
                        p_sum += pressure[i + 1, j]
            # 下
            if j > 0:
                if cell_type[i, j - 1] != CELL_SOLID:
                    n_neighbors += 1.0
                    if cell_type[i, j - 1] == CELL_FLUID:
                        p_sum += pressure[i, j - 1]
            # 上
            if j < GRID_RES - 1:
                if cell_type[i, j + 1] != CELL_SOLID:
                    n_neighbors += 1.0
                    if cell_type[i, j + 1] == CELL_FLUID:
                        p_sum += pressure[i, j + 1]

            if n_neighbors > 0:
                new_p = (p_sum + DX * DX * divergence[i, j] * RHO / DT) / n_neighbors
                residual += (new_p - pressure[i, j]) ** 2
                pressure[i, j] = new_p

    return residual


def solve_pressure():
    """压力求解主循环"""
    # 清零压力
    pressure.fill(0.0)

    # Jacobi 迭代
    for iteration in range(100):
        residual = pressure_jacobi_iteration()
        if residual < 1e-6:
            break


# ==================== 步骤 7: 压力投影 ====================
@ti.kernel
def project_velocity():
    """
    用压力梯度修正速度，使其无散度:
    u_new = u - (dt/ρ) * ∂p/∂x
    v_new = v - (dt/ρ) * ∂p/∂y

    投影后的速度场满足 ∇·u = 0（不可压缩条件）

    这就是 Niagara 中 "Project Pressure" stage 做的事
    """
    # 修正 u
    for i, j in u:
        if 1 <= i < GRID_RES and 0 <= j < GRID_RES:
            # 只修正流体-流体或流体-空气界面
            if cell_type[i - 1, j] == CELL_FLUID or cell_type[i, j] == CELL_FLUID:
                if cell_type[i - 1, j] == CELL_SOLID or cell_type[i, j] == CELL_SOLID:
                    u[i, j] = 0.0  # 固体边界
                else:
                    u[i, j] -= (DT / (RHO * DX)) * (pressure[i, j] - pressure[i - 1, j])

    # 修正 v
    for i, j in v:
        if 0 <= i < GRID_RES and 1 <= j < GRID_RES:
            if cell_type[i, j - 1] == CELL_FLUID or cell_type[i, j] == CELL_FLUID:
                if cell_type[i, j - 1] == CELL_SOLID or cell_type[i, j] == CELL_SOLID:
                    v[i, j] = 0.0
                else:
                    v[i, j] -= (DT / (RHO * DX)) * (pressure[i, j] - pressure[i, j - 1])


# ==================== 步骤 8: G2P (Grid to Particle) — FLIP 核心 ====================
@ti.kernel
def grid_to_particle():
    """
    FLIP 的灵魂所在:
    - PIC:  v_particle = v_grid_new              (完全从网格取，平滑但丢细节)
    - FLIP: v_particle += (v_grid_new - v_grid_old)  (只取增量，保留粒子细节但可能不稳定)
    - 混合: v = α*FLIP + (1-α)*PIC

    对比 SPH: SPH 没有这步，因为速度始终在粒子上，从未离开过
    这正是 FLIP 的"精髓"：借用网格做了压力求解，然后只把"修正量"还给粒子
    """
    for p in range(num_particles[None]):
        pos = particle_pos[p]

        # 从新网格插值得到 PIC 速度
        pic_u = bilinear_interp_u(pos)
        pic_v = bilinear_interp_v(pos)

        # 从旧网格插值 (P2G 之后、压力投影之前的速度)
        old_u = bilinear_interp_u_old(pos)
        old_v = bilinear_interp_v_old(pos)

        # FLIP 增量
        flip_u = particle_vel[p].x + (pic_u - old_u)
        flip_v = particle_vel[p].y + (pic_v - old_v)

        # PIC/FLIP 混合
        particle_vel[p].x = FLIP_RATIO * flip_u + (1.0 - FLIP_RATIO) * pic_u
        particle_vel[p].y = FLIP_RATIO * flip_v + (1.0 - FLIP_RATIO) * pic_v


# ==================== 步骤 9: 粒子位移 ====================
@ti.kernel
def advect_particles():
    """
    移动粒子 + 边界碰撞
    使用前向欧拉（可升级为 RK2）

    对比 SPH: 逻辑基本相同，但 SPH 的边界用弹簧力，这里用硬约束
    """
    for p in range(num_particles[None]):
        particle_pos[p] += particle_vel[p] * DT

        # 硬边界约束
        margin = DX * 1.5  # 留一点余量，不贴墙

        if particle_pos[p].x < margin:
            particle_pos[p].x = margin
            particle_vel[p].x = 0.0
        if particle_pos[p].x > DOMAIN_X - margin:
            particle_pos[p].x = DOMAIN_X - margin
            particle_vel[p].x = 0.0
        if particle_pos[p].y < margin:
            particle_pos[p].y = margin
            particle_vel[p].y = 0.0
        if particle_pos[p].y > DOMAIN_Y - margin:
            particle_pos[p].y = DOMAIN_Y - margin
            particle_vel[p].y = 0.0


# ==================== 初始化 ====================
def initialize_dam_break():
    """
    经典水坝崩塌场景：左半部分填满水
    每个格子放 2x2 = 4 个粒子（FLIP 标准做法）
    """
    positions = []
    velocities = []

    # 在左下方区域撒粒子 (水坝)
    particles_per_cell = 4  # 每个格子 2x2 粒子
    sub = int(np.sqrt(particles_per_cell))

    for i in range(2, GRID_RES // 3):
        for j in range(2, int(GRID_RES * 0.7)):
            for si in range(sub):
                for sj in range(sub):
                    px = (i + (si + 0.5) / sub) * DX
                    py = (j + (sj + 0.5) / sub) * DX
                    positions.append([px, py])
                    velocities.append([0.0, 0.0])

    n = len(positions)
    if n > MAX_PARTICLES:
        n = MAX_PARTICLES
        positions = positions[:n]
        velocities = velocities[:n]

    num_particles[None] = n
    particle_pos.from_numpy(np.array(positions[:n], dtype=np.float32))
    particle_vel.from_numpy(np.array(velocities[:n], dtype=np.float32))

    print(f"初始化完成: {n} 个粒子")
    print(f"网格分辨率: {GRID_RES}x{GRID_RES}")
    print(f"FLIP/PIC 比例: {FLIP_RATIO}")


# ==================== 单步模拟 ====================
def step():
    """
    完整的一帧模拟流程
    对应 Niagara 中从 Build Kernel Array 到 FLIP/PIC force 的全部 Stage
    """
    # 1. 标记单元格类型 (对应 Niagara: Compute Boundary)
    mark_cells()

    # 2. P2G: 粒子速度 → 网格 (对应 Niagara: Rasterize Particles)
    particle_to_grid()
    normalize_grid_velocity()

    # 3. 保存旧速度 (对应 Niagara: Initialize StartVelocity Grid)
    save_velocity()

    # 4. 施加重力 (外力)
    apply_gravity()

    # 5. 边界条件
    apply_boundary_conditions()

    # 6. 压力求解 (对应 Niagara: Compute Divergence + Solve Pressure)
    compute_divergence_field()
    solve_pressure()

    # 7. 压力投影 (对应 Niagara: Project Pressure)
    project_velocity()

    # 8. 再次边界条件（投影可能破坏边界）
    apply_boundary_conditions()

    # 9. G2P: 网格速度增量 → 粒子 (对应 Niagara: FLIP/PIC force)
    grid_to_particle()

    # 10. 粒子移动
    advect_particles()


# ==================== 可视化 ====================
def main():
    print("=" * 60)
    print("FLIP 流体解算器 (2D 水坝崩塌)")
    print("=" * 60)
    print()
    print("与你的 SPH 解算器的关键区别:")
    print("  SPH:  粒子直接交互 → 状态方程近似压力 → 容易压缩")
    print("  FLIP: 粒子→网格→泊松方程精确求压→增量回粒子 → 严格不可压")
    print()

    initialize_dam_break()

    # Matplotlib 实时可视化
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(0, DOMAIN_X)
    ax.set_ylim(0, DOMAIN_Y)
    ax.set_aspect('equal')
    ax.set_title('FLIP Fluid Simulation (Dam Break)')

    # 画边界
    boundary = Rectangle((DX, DX),
                          DOMAIN_X - 2 * DX, DOMAIN_Y - 2 * DX,
                          linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(boundary)

    scatter = ax.scatter([], [], s=1.5, c='dodgerblue', alpha=0.7)

    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         verticalalignment='top', fontsize=10)

    def animate(frame_num):
        # 每帧模拟 3 个子步
        for _ in range(3):
            step()

        pos = particle_pos.to_numpy()[:num_particles[None]]
        scatter.set_offsets(pos)

        # 用速度大小着色
        vel = particle_vel.to_numpy()[:num_particles[None]]
        speed = np.linalg.norm(vel, axis=1)
        scatter.set_array(speed)
        scatter.set_cmap('coolwarm')
        scatter.set_clim(0, 3.0)

        frame_text.set_text(f'Frame: {frame_num * 3}  Particles: {num_particles[None]}')
        return scatter, frame_text

    ani = animation.FuncAnimation(
        fig, animate,
        frames=None,
        interval=33,  # ~30 FPS
        blit=False,
        cache_frame_data=False
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
