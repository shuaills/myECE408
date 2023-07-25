

Here is the English version of the steps to resolve Nsight Compute permission issue on WSL2 in Markdown format:

# Resolving Nsight Compute Permission Issue on WSL2

## Problem

When using Nsight Compute to profile CUDA programs on WSL2, the following error occurs:

```
NVIDIA Development Tools Solutions - ERR_NVGPUCTRPERM: Permission issue with Performance Counters
```

## Cause 

By default, Windows restricts normal users from accessing the GPU performance counters.

## Solution

1. Open NVIDIA Control Panel on Windows, right click and select "Run as administrator".

2. In the "Desktop" tab of the control panel, make sure "Enable developer settings" is checked. 

3. Under "Developer" > "Manage GPU performance counters", select "Allow access to the GPU performance counters to all users" to grant unrestricted profiling permissions.

4. Restart Windows for the settings to take effect.

5. Now run Nsight Compute again on WSL2, it should be able to profile the GPU performance normally.

Reference: https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#AllUsersTag

### Problem

When using Nsight Compute, the following warning occurs:

```
==WARNING== Could not deploy stock section files to "/root/Documents/NVIDIA Nsight Compute/2021.3.1/Sections". Set the HOME environment variable to a writable directory.  
```

### Cause

The default Nsight Compute sections folder is not writable, unable to deploy the default sections files.

### Solution

Reset the sections folder using the command:

```
ncu --section-folder-restore
```

This will restore the default sections folder and files.

# 在 WSL2 中解决 Nsight Compute 的权限问题

## 问题

在 WSL2 中使用 Nsight Compute 分析 CUDA 程序时,遇到以下错误:

```
NVIDIA Development Tools Solutions - ERR_NVGPUCTRPERM: Permission issue with Performance Counters
```

## 原因

默认情况下,Windows 限制了普通用户访问 GPU 性能计数器的权限。

## 解决方法

1. 在 Windows 中打开 NVIDIA 控制面板,右键选择“以管理员身份运行”。

2. 在控制面板的“桌面”选项卡中,确保勾选了“启用开发人员设置”。

3. 在“开发人员”>“管理 GPU 性能计数器”中,选择“允许所有用户访问 GPU 性能计数器”,以授予不受限制的分析权限。

4. 重启 Windows 使设置生效。

5. 现在在 WSL2 中重新运行 Nsight Compute,就可以正常进行 GPU 性能分析了。

参考链接：https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#AllUsersTag




### 问题

在使用 Nsight Compute 时,出现以下警告:

```
==WARNING== Could not deploy stock section files to "/root/Documents/NVIDIA Nsight Compute/2021.3.1/Sections". Set the HOME environment variable to a writable directory.
```

### 原因

默认的 Nsight Compute 分析区文件夹不可写,无法部署默认的分析区文件。

### 解决方法

使用以下命令重置分析区文件夹:

```
ncu --section-folder-restore
```

这将恢复默认的分析区文件夹和文件。