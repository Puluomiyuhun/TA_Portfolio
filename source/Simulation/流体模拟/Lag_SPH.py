import taichi as ti
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 初始化Taichi
ti.init(arch=ti.gpu, default_fp=ti.f32)

# ==================== 配置参数 ====================
N = 6000  # 粒子数

# 3D模拟空间
SIM_W = 0.5    # X和Y方向的宽度
SIM_H = 1.0    # Z方向高度

# 物理参数
SPACING = 0.05        # 粒子间距
R = SPACING * 1.5     # 邻居半径
R_SQ = R * R
G = ti.Vector([0.0, 0.0, -0.02 * 0.5])  # 减小重力
REST_DENSITY = 2    # 减小静止密度
K = SPACING / 1000.0   # 增大压力系数（让流体更"硬"）
K_NEAR = K * 3        # 减小近距离压力系数
SIGMA = 0.05          # 增大粘度系数
MAX_VEL = 1.0        # 减小最大速度
WALL_DAMP = 0.3      # 增大墙壁阻尼（更软）
VEL_DAMP = 0.7       # 减小速度阻尼
dt = 0.25            # 显著减小时间步长

# ==================== 网格索引参数 ====================
GRID_SIZE = R * 1.2  # 稍大一点确保包含所有邻居
GRID_SIZE_INV = 1.0 / GRID_SIZE

GRID_NX = int(2 * SIM_W / GRID_SIZE) + 3
GRID_NY = int(2 * SIM_W / GRID_SIZE) + 3
GRID_NZ = int(SIM_H / GRID_SIZE) + 3
TOTAL_GRIDS = GRID_NX * GRID_NY * GRID_NZ

print(f"网格配置: {GRID_NX} × {GRID_NY} × {GRID_NZ} = {TOTAL_GRIDS:,} 个网格")
print(f"时间步长 dt = {dt}")
print(f"最大速度 MAX_VEL = {MAX_VEL}")

# ==================== Taichi 3D字段定义 ====================
# 粒子属性
particle_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
particle_vel = ti.Vector.field(3, dtype=ti.f32, shape=N)
particle_force = ti.Vector.field(3, dtype=ti.f32, shape=N)
particle_density = ti.field(dtype=ti.f32, shape=N)
particle_density_near = ti.field(dtype=ti.f32, shape=N)
particle_pressure = ti.field(dtype=ti.f32, shape=N)
particle_pressure_near = ti.field(dtype=ti.f32, shape=N)

# 网格索引系统
grid_particles_count = ti.field(dtype=ti.i32, shape=TOTAL_GRIDS)
grid_particles_offset = ti.field(dtype=ti.i32, shape=TOTAL_GRIDS)
grid_particles_ids = ti.field(dtype=ti.i32, shape=(TOTAL_GRIDS, 64))  # 减少每个网格最大粒子数

# 辅助数组
particle_grid_index = ti.field(dtype=ti.i32, shape=N)
sorted_particle_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
sorted_particle_vel = ti.Vector.field(3, dtype=ti.f32, shape=N)
sorted_particle_density = ti.field(dtype=ti.f32, shape=N)
sorted_particle_density_near = ti.field(dtype=ti.f32, shape=N)
sorted_particle_pressure = ti.field(dtype=ti.f32, shape=N)
sorted_particle_pressure_near = ti.field(dtype=ti.f32, shape=N)

# 可视化
visual_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)

# ==================== 网格索引工具函数 ====================
@ti.func
def pos_to_grid_index(pos: ti.math.vec3) -> ti.i32:
    """将3D位置转换为网格索引"""
    grid_x = ti.cast((pos.x + SIM_W) * GRID_SIZE_INV + 1, ti.i32)  # +1 避免边界问题
    grid_y = ti.cast((pos.y + SIM_W) * GRID_SIZE_INV + 1, ti.i32)
    grid_z = ti.cast(pos.z * GRID_SIZE_INV + 1, ti.i32)
    
    grid_x = ti.max(0, ti.min(grid_x, GRID_NX - 1))
    grid_y = ti.max(0, ti.min(grid_y, GRID_NY - 1))
    grid_z = ti.max(0, ti.min(grid_z, GRID_NZ - 1))
    
    return grid_z * GRID_NX * GRID_NY + grid_y * GRID_NX + grid_x

@ti.func
def grid_index_to_xyz(idx: ti.i32):
    """将线性网格索引转换为3D网格坐标"""
    grid_x = idx % GRID_NX
    grid_y = (idx // GRID_NX) % GRID_NY
    grid_z = idx // (GRID_NX * GRID_NY)
    return grid_x, grid_y, grid_z

@ti.func
def get_neighbor_grids(idx: ti.i32):
    """获取邻居网格索引"""
    neighbors = ti.Vector([0] * 27, dt=ti.i32)
    count = 0
    
    gx, gy, gz = grid_index_to_xyz(idx)
    
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ngx = gx + dx
                ngy = gy + dy
                ngz = gz + dz
                
                if 0 <= ngx < GRID_NX and 0 <= ngy < GRID_NY and 0 <= ngz < GRID_NZ:
                    neighbor_idx = ngz * GRID_NX * GRID_NY + ngy * GRID_NX + ngx
                    neighbors[count] = neighbor_idx
                    count += 1
    
    return neighbors, count

# ==================== 修正的核函数和力计算 ====================
@ti.func
def smoothing_kernel(r: ti.f32, h: ti.f32) -> ti.f32:
    """修复的核函数 - 使用ti.select避免if返回"""
    q = r / h
    # 使用ti.select代替if-return
    return ti.select(q < 1.0, (1.0 - q) * (1.0 - q), 0.0)

@ti.func
def smoothing_kernel_gradient(r: ti.f32, h: ti.f32) -> ti.f32:
    """核函数梯度"""
    q = r / h
    # 使用ti.select
    return ti.select((q > 0.0) & (q < 1.0), -2.0 * (1.0 - q) / h, 0.0)

# ==================== 网格构建内核 ====================
@ti.kernel
def build_grid():
    """构建网格索引"""
    # 重置网格计数
    for i in range(TOTAL_GRIDS):
        grid_particles_count[i] = 0
    
    # 统计每个网格中的粒子数
    for i in range(N):
        pos = particle_pos[i]
        grid_idx = pos_to_grid_index(pos)
        ti.atomic_add(grid_particles_count[grid_idx], 1)
        particle_grid_index[i] = grid_idx
    
    # 计算前缀和
    offset = 0
    for i in range(TOTAL_GRIDS):
        grid_particles_offset[i] = offset
        offset += grid_particles_count[i]
    
    # 重置计数用于填充
    for i in range(TOTAL_GRIDS):
        grid_particles_count[i] = 0
    
    # 将粒子分配到网格
    for i in range(N):
        grid_idx = particle_grid_index[i]
        local_idx = ti.atomic_add(grid_particles_count[grid_idx], 1)
        if local_idx < 64:  # 确保不超过数组大小
            grid_particles_ids[grid_idx, local_idx] = i

# ==================== 修正的密度计算 ====================
@ti.kernel
def compute_density():
    """修正的密度计算"""
    # 重置密度
    for i in range(N):
        particle_density[i] = 0.0
        particle_density_near[i] = 0.0
    
    # 使用网格加速
    for i in range(N):
        pos_i = particle_pos[i]
        grid_idx = pos_to_grid_index(pos_i)
        neighbor_grids, neighbor_count = get_neighbor_grids(grid_idx)
        
        for n in range(neighbor_count):
            ng_idx = neighbor_grids[n]
            count = grid_particles_count[ng_idx]
            
            for local_idx in range(count):
                j = grid_particles_ids[ng_idx, local_idx]
                if i != j:
                    pos_j = particle_pos[j]
                    r_vec = pos_i - pos_j
                    r_sq = r_vec.norm_sqr()
                    
                    if r_sq < R_SQ:
                        r = ti.sqrt(r_sq)
                        q = r / R
                        
                        # 使用更稳定的核函数
                        w = smoothing_kernel(r, R)
                        w_near = w * (1.0 - q)  # 近距离权重
                        
                        particle_density[i] += w
                        particle_density_near[i] += w_near
                        
                        ti.atomic_add(particle_density[j], w)
                        ti.atomic_add(particle_density_near[j], w_near)

# ==================== 修正的压力计算 ====================
@ti.kernel
def compute_pressure():
    """修正的压力计算"""
    for i in range(N):
        # 添加最小密度限制，避免负压过大
        density = ti.max(particle_density[i], 0.5 * REST_DENSITY)
        particle_pressure[i] = K * (density - REST_DENSITY)
        
        # 限制压力范围，避免过大
        
        particle_pressure_near[i] = K_NEAR * particle_density_near[i]

# ==================== 修正的力计算 ====================
@ti.kernel
def compute_forces():
    """修正的力计算 - 更稳定的版本"""
    # 重置力（只保留重力）
    for i in range(N):
        particle_force[i] = G
    
    # 使用网格加速
    for i in range(N):
        pos_i = particle_pos[i]
        vel_i = particle_vel[i]
        p_i = particle_pressure[i]
        p_near_i = particle_pressure_near[i]
        
        grid_idx = pos_to_grid_index(pos_i)
        neighbor_grids, neighbor_count = get_neighbor_grids(grid_idx)
        
        for n in range(neighbor_count):
            ng_idx = neighbor_grids[n]
            count = grid_particles_count[ng_idx]
            
            for local_idx in range(count):
                j = grid_particles_ids[ng_idx, local_idx]
                if i < j:
                    pos_j = particle_pos[j]
                    r_vec = pos_j - pos_i
                    r_sq = r_vec.norm_sqr()
                    
                    if 0.0 < r_sq < R_SQ:
                        r = ti.sqrt(r_sq)
                        q = r / R
                        
                        # 邻居属性
                        vel_j = particle_vel[j]
                        p_j = particle_pressure[j]
                        p_near_j = particle_pressure_near[j]
                        
                        # 核函数值
                        w = smoothing_kernel(r, R)
                        grad_w = smoothing_kernel_gradient(r, R)
                        
                        if grad_w != 0.0:
                            # 修正的压力力计算
                            # 使用对称化公式：F = -V_i V_j (p_i/ρ_i² + p_j/ρ_j²) ∇W
                            # 这里简化为：F = -(p_i + p_j) * ∇W
                            pressure_term = (p_i + p_j) + (p_near_i + p_near_j) * (1.0 - q)
                            
                            # 归一化方向向量
                            r_norm = r_vec / r
                            
                            # 压力力（方向修正）
                            f_pressure = -pressure_term * grad_w * r_norm
                            
                            # 粘性力
                            vel_rel = vel_i - vel_j
                            vel_dot = vel_rel.dot(r_norm)
                            
                            f_visc = ti.Vector([0.0, 0.0, 0.0])
                            if vel_dot > 0.0:
                                # 使用更稳定的粘性公式
                                mu = (r * vel_dot) / (r_sq + 0.01 * R * R)
                                f_visc = SIGMA * mu * r_norm * w
                            
                            # 总力（限制大小）
                            f_total = f_pressure + f_visc
                            
                            # 限制力的大小，避免爆炸
                            f_mag = f_total.norm()
                            max_force = 10.0 * MAX_VEL / dt
                            if f_mag > max_force:
                                f_total = f_total * (max_force / f_mag)
                            
                            # 应用力
                            ti.atomic_add(particle_force[i], -f_total)
                            ti.atomic_add(particle_force[j], f_total)

# ==================== 修正的边界条件 ====================
@ti.kernel
def apply_boundary_conditions():
    """更温和的边界条件"""
    for i in range(N):
        pos = particle_pos[i]
        vel = particle_vel[i]
        force = particle_force[i]
        
        # X方向边界
        if pos.x < -SIM_W:
            penetration = -SIM_W - pos.x
            # 软边界：力与穿透深度成正比
            force.x += WALL_DAMP * penetration
            # 轻微的速度反弹
            if vel.x < 0:
                vel.x = -vel.x * 0.3
        
        if pos.x > SIM_W:
            penetration = pos.x - SIM_W
            force.x -= WALL_DAMP * penetration
            if vel.x > 0:
                vel.x = -vel.x * 0.3
        
        # Y方向边界
        if pos.y < -SIM_W:
            penetration = -SIM_W - pos.y
            force.y += WALL_DAMP * penetration
            if vel.y < 0:
                vel.y = -vel.y * 0.3
        
        if pos.y > SIM_W:
            penetration = pos.y - SIM_W
            force.y -= WALL_DAMP * penetration
            if vel.y > 0:
                vel.y = -vel.y * 0.3
        
        # Z方向边界（地面）
        if pos.z < 0.0:
            penetration = -pos.z
            force.z += WALL_DAMP * penetration
            if vel.z < 0:
                vel.z = -vel.z * 0.2  # 地面反弹更小
        
        # Z方向边界（天花板）
        if pos.z > SIM_H:
            penetration = pos.z - SIM_H
            force.z -= WALL_DAMP * penetration
            if vel.z > 0:
                vel.z = -vel.z * 0.3
        
        particle_vel[i] = vel
        particle_force[i] = force

# ==================== 修正的粒子更新 ====================
@ti.kernel
def update_particles():
    """更稳定的粒子更新"""
    for i in range(N):
        # 半隐式欧拉方法：先更新速度，再更新位置
        # v_{n+1} = v_n + a_n * Δt
        particle_vel[i] += particle_force[i] * dt
        
        # 严格的速度限制
        speed = particle_vel[i].norm()
        if speed > MAX_VEL:
            particle_vel[i] *= MAX_VEL / speed
        
        # x_{n+1} = x_n + v_{n+1} * Δt
        particle_pos[i] += particle_vel[i] * dt

# ==================== 初始化 ====================
@ti.kernel
def initialize_particles():
    """初始化3D粒子排列"""
    # 计算立方体排列
    n_per_dim = int(ti.cast(N, ti.f32) ** (1.0/3.0)) + 1
    spacing = SPACING
    
    for idx in range(N):
        # 将索引转换为3D坐标
        i = idx // (n_per_dim * n_per_dim)
        j = (idx // n_per_dim) % n_per_dim
        k = idx % n_per_dim
        
        # 水坝后方区域
        x = -SIM_W * 0.7 + i * spacing * 0.8
        y = -SIM_W * 0.3 + j * spacing * 0.8
        z = 0.1 + k * spacing * 0.8
        
        particle_pos[idx] = ti.Vector([x, y, z])
        visual_pos[idx] = ti.Vector([x, y, z])

# ==================== 主程序 ====================
def main():
    print("3D SPH模拟 - 稳定版本")
    print(f"粒子数量: {N:,}")
    print(f"时间步长: {dt}")
    print(f"最大速度: {MAX_VEL}")
    
    # 初始化粒子
    print("初始化粒子...")
    initialize_particles()
    
    # 设置3D图形
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(-SIM_W, SIM_W)
    ax.set_ylim(-SIM_W, SIM_W)
    ax.set_zlim(0, SIM_H)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=25, azim=45)
    
    # 初始散点图
    scatter = ax.scatter([], [], [], 
                        s=200,
                        alpha=0.7,
                        edgecolors='none',
                        depthshade=True)
    
    frame = 0

    # 预分配颜色数组
    colors_cache = np.zeros((N, 3))
    
    def animate(frame_num):
        nonlocal frame
        
        # 稳定的SPH步骤
        build_grid()
        compute_density()
        compute_pressure()
        compute_forces()
        apply_boundary_conditions()
        update_particles()
        
        # 获取位置用于可视化
        positions_np = particle_pos.to_numpy()

        # 计算颜色（基于高度或压力）
        pressures_np = np.abs(particle_pressure.to_numpy())
        if np.max(pressures_np) > 0:
            # 根据压力着色
            normalized_pressure = pressures_np / np.max(pressures_np)
            normalized_pressure = np.clip(normalized_pressure, 0, 1)
            
            # 从蓝色（低压）到红色（高压）
            colors_cache[:, 0] = normalized_pressure  # R
            colors_cache[:, 1] = 0.0                  # G
            colors_cache[:, 2] = 1.0 - normalized_pressure  # B
        
        # 更新图形
        scatter._offsets3d = (positions_np[:, 0], 
                             positions_np[:, 1], 
                             positions_np[:, 2])
        scatter.set_color(colors_cache)
        frame += 1
        
        return scatter,
    
    # 创建动画
    print("开始模拟...")
    ani = animation.FuncAnimation(
        fig, animate,
        frames=None,
        interval=0.01,  # 更慢的更新速度，便于观察
        blit=False,
        cache_frame_data=False
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        # 尝试更保守的参数
        print("\n尝试更保守的参数...")
        N = 4000
        dt = 0.05
        MAX_VEL = 0.3
        main()