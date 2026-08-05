# 在 Windows host-only 网络部署 Origin Worker

Origin Worker 只能绑定 Windows 主机与 Linux 虚拟机共享的 host-only 网卡，不应绑定
`0.0.0.0`、公网网卡、回环地址或链路本地地址。以下示例假定：

- Windows host-only 地址：`192.168.56.1`
- Linux 虚拟机地址：`192.168.56.2`
- host-only 子网：`192.168.56.0/24`
- Worker 端口：`8443`

## 部署配置

1. 为 `192.168.56.1` 签发本地自签名服务器证书，把私钥仅留在 Windows，并把证书副本安全复制到 Linux。证书必须包含 Worker 使用的 DNS 名或 IP SAN。
2. 在部署系统中生成至少 32 个随机字符的 Bearer Token，通过
   `ORIGIN_WORKER_TOKEN` 注入 Windows 服务。不要把 Token、私钥或证书内容写入仓库、命令历史、审计事件或 Fit Archive。
3. 使用 Windows 服务专属的本地 NTFS 目录保存 `--state-dir`。该目录不能是符号链接、UNC 路径或共享目录，并且只授予服务账户访问权限。
4. 以管理员 PowerShell 仅允许 Linux 虚拟机访问 host-only 地址：

   ```powershell
   New-NetFirewallRule `
     -DisplayName "Origin Worker from Linux guest" `
     -Direction Inbound -Action Allow -Protocol TCP `
     -LocalAddress 192.168.56.1 -LocalPort 8443 `
     -RemoteAddress 192.168.56.2
   ```

   同时确认没有允许公网网卡或任意远端地址访问该端口的更宽规则。

5. 启动 Worker：

   ```powershell
   origin-worker serve `
     --state-dir C:\ProgramData\OriginWorker `
     --host 192.168.56.1 `
     --host-only-network 192.168.56.0/24 `
     --linux-guest-address 192.168.56.2 `
     --port 8443 `
     --certfile C:\ProgramData\OriginWorker\tls\worker.crt `
     --keyfile C:\ProgramData\OriginWorker\tls\worker.key
   ```

   生产模式默认启动专用 OriginPro Adapter。只有诊断桌面自动化时才加
   `--origin-visible`；`--fake-origin` 仅用于开发与回归测试，不能用于生产服务，且不能和
   `--origin-visible` 同时使用。

`--host-only-network` 是部署者对虚拟网络边界的明确声明；Worker 不会尝试从
Windows 网卡名称或类型自动推断 host-only 网络。`--linux-guest-address` 必须是该子网内、
不同于 Worker 绑定地址的私有单播地址，并且必须与上面防火墙规则的
`-RemoteAddress` 完全一致。除 Windows 防火墙限制外，Worker 还会在每个 `/v1`
请求进入鉴权和业务处理前，只允许这个来源地址；无法取得客户端地址时也会拒绝请求。

`serve` 会在开始接收请求前验证 host-only 绑定与 Linux guest 地址、证书和私钥文件、
至少 32 字符的 Token、本地工作目录、SQLite 可读写性与 Adapter 配置。任何预检失败都会让进程退出。

## Linux 端证书固定

Linux 应只使用 HTTPS URL，并把复制来的自签名证书作为固定信任根：

```python
transport = HttpWorkerTransport.with_pinned_certificate(
    "https://192.168.56.1:8443",
    token=worker_token,
    pinned_certificate=Path("/etc/origin-worker/worker.crt"),
)
```

不要关闭 TLS 校验，也不要把系统公共 CA 集合替代为该固定证书。Token 应由 Linux 的部署秘密配置提供。

## 运行与恢复

- SQLite 保存队列与状态转换，工作区保存 Dataset Snapshot、受控诊断和 Fit Result Bundle。不要手工修改数据库或工作区引用。
- Worker 重启会把遗留的 `running` Fit Job 标记为 `worker_restarted` 失败，保留 `queued` Fit Job，且不会自动重跑失败任务。
- 运行中取消或默认 30 分钟硬超时会终止当前执行实例；后续排队任务使用干净实例继续。
- 终态工作区默认保留 7 天。定时清理只删除 Windows Worker 工作区，不删除已经复制并校验到 Linux 的 Fit Archive。
- 运维日志与审计事件只能记录对象 ID、状态和安全错误码。原始数据、Token、私钥、完整异常文本和内部栈不得进入审计流。
