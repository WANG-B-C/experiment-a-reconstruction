
import numpy as np
import matplotlib.pyplot as plt


def generate_data(n=1000, seed=42):
    """生成 n 个三维隐藏状态。"""
    rng = np.random.default_rng(seed)
    Z = rng.random((n, 3))
    return Z


def observe(Z):
    """将每个三维状态拆成三个二维视图。"""
    view_ab = Z[:, [0, 1]]
    view_ac = Z[:, [0, 2]]
    view_bc = Z[:, [1, 2]]

    return view_ab, view_ac, view_bc


def reconstruct(view_ab, view_ac, view_bc):
    """根据已知坐标对应关系重建三维状态。"""
    a = view_ab[:, 0]
    b = view_ab[:, 1]
    c = view_ac[:, 1]

    Z_reconstructed = np.column_stack([a, b, c])
    return Z_reconstructed


def evaluate(Z, Z_reconstructed):
    """计算重建误差。"""
    errors = np.linalg.norm(
        Z - Z_reconstructed,
        axis=1
    )

    print(f"样本数量: {len(Z)}")
    print(f"最大误差: {errors.max():.12f}")
    print(f"平均误差: {errors.mean():.12f}")
    print(f"误差标准差: {errors.std():.12f}")

    return errors


def visualize(Z, Z_reconstructed):
    """比较原始点与重建点。"""
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(
        Z[:, 0], Z[:, 1], Z[:, 2],
        s=10, alpha=0.6
    )
    ax1.set_title("Original 3D states")
    ax1.set_xlabel("a")
    ax1.set_ylabel("b")
    ax1.set_zlabel("c")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(
        Z_reconstructed[:, 0],
        Z_reconstructed[:, 1],
        Z_reconstructed[:, 2],
        s=10, alpha=0.6
    )
    ax2.set_title("Reconstructed 3D states")
    ax2.set_xlabel("a")
    ax2.set_ylabel("b")
    ax2.set_zlabel("c")

    plt.tight_layout()
    plt.show()


def main():
    # 1. 生成隐藏状态
    Z = generate_data(n=1000)

    # 2. 获取三个不完整视图
    view_ab, view_ac, view_bc = observe(Z)

    # 检查三个视图中的重复坐标是否一致
    a_error = np.abs(view_ab[:, 0] - view_ac[:, 0])
    b_error = np.abs(view_ab[:, 1] - view_bc[:, 0])
    c_error = np.abs(view_ac[:, 1] - view_bc[:, 1])

    print("a 的最大一致性误差:", a_error.max())
    print("b 的最大一致性误差:", b_error.max())
    print("c 的最大一致性误差:", c_error.max())
    
    # 3. 重建隐藏状态
    Z_reconstructed = reconstruct(
        view_ab, view_ac, view_bc
    )

    # 实验 B0：只保留两个视图
    Z_partial = np.column_stack([
        view_ab[:, 0],  # a
        view_ab[:, 1],  # b
        view_ac[:, 1]   # c
    ])

    # 注意：这里虽然没有使用 view_bc，
    # 但 view_ab 和 view_ac 已经覆盖了 a、b、c 三个坐标。
    errors_partial = np.linalg.norm(
        Z - Z_partial, axis=1
    )

    print("\n只用 view_ab 和 view_ac：")
    print("最大误差:", errors_partial.max())
    print("平均误差:", errors_partial.mean())

    # 4. 评估重建质量
    evaluate(Z, Z_reconstructed)

    # 5. 可视化
    visualize(Z, Z_reconstructed)

    # 实验 B1：只知道 a、b，不知道 c
    Z_incomplete = np.column_stack([
        view_ab[:, 0],
        view_ab[:, 1],
        np.zeros(len(view_ab))
    ])

    errors_incomplete = np.linalg.norm(
        Z - Z_incomplete, axis=1
    )

    print("\n只用 view_ab，并把未知 c 暂时设为 0：")
    print("最大误差:", errors_incomplete.max())
    print("平均误差:", errors_incomplete.mean())

    
    # 两种不同的缺失坐标估计方法
    c_true = Z[:, 2]

    # 方法 1：把未知 c 估计为 0
    c_pred_zero = np.zeros(len(Z))

    # 方法 2：利用已知分布，把未知 c 估计为 0.5
    c_pred_mean = np.full(len(Z), 0.5)

    for name, c_pred in [
        ("猜测 c = 0", c_pred_zero),
        ("猜测 c = 0.5", c_pred_mean),
    ]:
        abs_errors = np.abs(c_true - c_pred)
        mse = np.mean((c_true - c_pred) ** 2)
        rmse = np.sqrt(mse)

        print(f"\n{name}")
        print(f"平均绝对误差: {abs_errors.mean():.4f}")
        print(f"均方根误差: {rmse:.4f}")

if __name__ == "__main__":
    main()



    rng = np.random.default_rng(42)
    n = 10000

    # a、b 是随机观测
    a = rng.random(n)
    b = rng.random(n)

    # 情况 1：c 独立于 a、b
    c_independent = rng.random(n)

    # 情况 2：c 完全由 a、b 决定
    c_structured = a + b

    # 两种情况下都只用 a、b 的信息预测 c
    c_pred = np.full(n, 0.5)

    mse_independent = np.mean((c_independent - c_pred) ** 2)

    # 已知结构规则时，直接恢复
    c_reconstructed = a + b
    mse_structured = np.mean((c_structured - c_reconstructed) ** 2)

    print("独立 c 的预测 MSE:", mse_independent)
    print("有结构时的重建 MSE:", mse_structured)


