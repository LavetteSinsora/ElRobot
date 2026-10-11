# g1d_deploy

在 Unitree G1-D 真机上运行策略的部署框架。

## 简介

策略输出末端位姿（eef）或关节角度，g1d_deploy 负责把这些动作在 G1-D 上执行出来。框架涵盖：

- 获取观测：对齐相机画面和关节状态的时间戳
- 执行策略的输出：
    - 逆运动学（IK）：把目标 eef 位姿转换成关节角度
    - 轨迹平滑和 500 Hz 伺服：让 G1-D 平滑地到达目标关节角度
- rollout 的记录和回看

策略通过一个适配器类接入，框架不限制策略的形式：可以是 VLA / WAM，可以是脚本或 code-as-policy，也可以是外部规划器，比如 GPT-6 Astra 生成的路径点。

目前仅支持装配 Dex-1 夹爪的 G1-D。

## 使用步骤

1. **安装**：运行 `scripts/install.sh`，然后 `source .venv/bin/activate`。详见 `docs/setup/install.md`。
2. **准备 `robot.toml`**，描述所使用的 G1-D 的配置：相机、末端执行器类型、机载电脑的 IP 地址等。
    - 常见的机器人配置（如 G1-D + Dex-1 + 鱼眼腕部相机）在 `src/g1d_deploy/robot/profiles/` 中，若硬件配置一致可直接使用。
3. **准备 `machine.toml`**（可选），描述所使用的部署机器（即执行策略推理、下发控制指令的机器）。
    - 用于指定 rollout 数据的存放目录、控制面板使用的端口等。
4. **编写 `policy.py`**，即策略适配器：把框架给出的观测转换成策略的输入，把策略的输出转换成框架规定的动作格式。
5. **编写 `deploy.toml`**，即一次部署的参数：使用哪个机器人配置和策略适配器、checkpoint 路径、控制频率、同步 / 异步推理、重规划间隔、IK 求解器等。

第 1–3 步可以使用 `skills/setup-g1d-deploy/`，第 4、5 步可以使用 `skills/deploy-a-policy/`。

## 运行

```bash
g1d_deploy check-policy path/to/deploy.toml   # 离线检查，不连接机器人
g1d_deploy run path/to/deploy.toml            # 启动
```

启动之后，g1d_deploy 依次检查相机视频流和关节状态的接收，把机械臂移动到 home 姿态，同时加载并预热策略。

准备就绪后，打开控制面板 `localhost:8080`（端口可在 `machine.toml` 中修改）：

- Start 开始一次 rollout，Reset 结束 rollout 并回到 home 姿态，Exit 退出。
- 两次 rollout 之间可以修改 prompt 和部署参数（控制频率、同步 / 异步、重规划间隔、IK 求解器、采样步数等），下一次 rollout 生效，不需要重启。

部署参数的含义见 `docs/deploying_a_policy/deploy_toml.md`。

## rollout 回看

g1d_deploy 会记录每次 rollout 的全部数据：相机画面、关节状态、策略输出的动作、下发的指令。默认存放在 `deploy.toml` 同目录下的 `rollout_record/`，也可以在 `machine.toml` 中指定统一的存放目录。

```bash
g1d_deploy inspect path/to/records    # 打开 localhost:8770
```

页面上可以拖动时间轴回放一次 rollout，相机画面和 3D 机器人同步显示。

回放之外，页面顶部的 **compare** 可以用来比较不同 checkpoint / IK 求解器在回放的 rollout 上的表现：

- **policy checkpoint 模式**：把录下的观测交给另一个 checkpoint 重新推理（开环），查看不同 checkpoint 在同样的观测下会给出怎样的动作。也可以更换 prompt 或采样步数。
- **ik 模式**：用其他 IK 求解器重新求解录下的目标位姿，对比各求解器的误差。

详见 `docs/deploying_a_policy/rollout_inspection.md`。

## 更换模块

g1d_deploy 为以下模块提供了统一接口，可以替换成自己的实现而不改框架代码：IK 求解器、500 Hz 平滑器、异步推理时动作块的融合方式。说明在 `docs/infra_logic/`，例子在 `examples/`。

## 文档

从 `docs/overview.md` 开始，里面有全部文档的索引。

## 后续计划

- 支持 BrainCo 灵巧手
