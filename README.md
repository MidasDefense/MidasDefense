# What?

Midas is the first adaptive and efficient safeguard framework for Linux-based IoT devices with the real-time behavior auditing mechanism.

# Project Directory Structure

```shell
Midas_Project
├── midas_binaries    #binaries of Midas in different architecture
│   ├── midas_arm
│   ├── midas_mips
│   └── midas_mipsel
├── policy		      #compiled MD policy, i.e. policy.31 and file_contexts file
│   ├── policy.31
|   └── file_contexts
└── supplementary-materials  
	├── physical-devices-with-Midas  #firmware files for physical devices and related experiment pictures
	├── comparison_experiment  #details about the quantitive comparison experiment
	├── Sensitive_Behavior_Acquisition.pdf  #details about the sensitive behavior acquisition procedure
	└── IoT_Malware_Dataset_Link.txt 		#The link of our IoT malware benchmark dateset
```



# Install Instructions

### 1. Compile firmware

First you need to create your own custom firmware. We recommand [OpenWrt](https://openwrt.org/) here, which is an open-source project for embedded operating systems based on Linux, primarily used on embedded devices. This frees you from the application selection and configuration provided by the vendor and allows you to customize the device through the use of packages to suit any application. For users, this means the ability for full customization, to use the device in ways never envisioned.

You can compile firmware of almost any architecture for a wide range of [IoT devices](https://openwrt.org/toh/start).

### 2. Enable SELinux

For OpenWrt, we recommend to try it like this:

```
Global build settings ->
[*] Enable SELinux
    default SELinux type (dssp) ->
```

Other configuration changes are not neccessary here, and all the SELinux-related configuration variants are automatically enabled.

### 3. Install Midas and MD Policy

- Running the compiled firmware QEMU or SoC/target supported by your firmware.
- Install Midas
  - select and move the suitable binary of midas to the firmware, like path /usr/sbin
  - run Midas as a background service or auto-execute after each boot.
- Install MD Policy
  - move the compiled binary MD policy, i.e. *poilcy.31*, to the SELinux subdirectory, e.g. /etc/selinux/selinux-policy/policy
  - move the *file_contexts* file to the SELinux subdirectory, e.g. /etc/selinux/selinux-policy/contexts/files
- Safeguard by Midas
  - load MD policy by the command `load_policy`
  - restart the “/etc/init.d/log” by the command `/etc/init.d/log restart`
  - you can add these command to “/etc/rc.local” to be executed once the system init finished.
- Finish