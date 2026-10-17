# BeamAdapter v23.06 — Case 1 修复补丁

本目录包含针对 SOFA **BeamAdapter v23.06** 插件中
`InterventionalRadiologyController` 的力学发散修复补丁。

## 背景

`InterventionalRadiologyController::applyInterventionalRadiologyController()` 的
STEP 3 在重插值节点位置时，会判断当前曲线横坐标 `xCurvAbs` 是否超过上一状态尖端
`prev_maxCurvAbs + threshold`。若超过，原代码走一个 **空的 Case 1 分支**——只打印
`msg_error`，不对该节点的位置做任何赋值。

当训练过程中器械推进较快、出现 `totalLengthIsChanging` 时，Case 1 会被触发，导致：

- 该节点位置保持未初始化/旧值，几何发生发散；
- 下游 LCP 求解器数值爆炸，worker 反复 timeout；
- 训练进度卡死（曾卡在 50%）。

## 修复内容

1. `msg_error` → `msg_warning`（Case 1 不再刷错误日志）。
2. 移除 `else`，让 Case 1 命中的节点 **fall-through** 到下方统一的插值逻辑，
   而非留在空分支。
3. 新增对 `prev_xId` 的 **clamp**：当 `prev_xId >= m_nodeCurvAbs.size()` 时钳到
   最后一个（尖端）节点，使 `baryCoef > 1`，spline 沿尖端方向外推。

## 应用方式

在 BeamAdapter v23.06 源码根目录执行：

```bash
git apply patches/BeamAdapter_Case1_fix.patch
# 或
patch -p1 < patches/BeamAdapter_Case1_fix.patch
```

重新编译后，用生成的 `libBeamAdapter.so.1.0` 替换 SOFA 发行版下的
`plugins/BeamAdapter/lib/libBeamAdapter.so.1.0`。

## 效果

修复后 success 率由 25% 提升至 62.5%，并消除了 worker 反复 timeout、几何发散导致
的进度卡死问题。