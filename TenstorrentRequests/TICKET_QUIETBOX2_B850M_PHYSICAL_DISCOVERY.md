# Support Ticket / Feature Request: Add ASUS B850M-C to TT-Metal Motherboard Discovery

- **Component**: TT-Fabric Physical System Discovery
- **File**: `tt_metal/fabric/physical_system_discovery.cpp`
- **Target Hardware**: Tenstorrent QuietBox 2 (4x Blackhole PCIe processors)
- **Severity**: Low / Cosmetic (Fallback works correctly, but emits recurring LogAlways warnings)
- **Reported Date**: 2026-09-28

---

## 1. Summary

During device initialization and mesh discovery on the Tenstorrent QuietBox 2 workstation, TT-Metal logs the following warning on every process startup:

```text
Unknown motherboard 'B850M-C' for chip_id=2 (bus_id=0x3) — falling back to bus_id as tray_id. Add this motherboard and its bus IDs to mobo_to_bus_ids in physical_system_discovery.cpp. (physical_system_discovery.cpp:119)
```

The system automatically falls back to using the PCI bus ID directly as the tray ID (`return TrayID{static_cast<uint32_t>(bus_id)};`). Because the Blackhole accelerators on the QuietBox 2 are mapped to PCI buses `0x01`, `0x02`, `0x03`, and `0x04`, the resulting tray assignments (`1, 2, 3, 4`) are functionally correct. However, the recurring warning causes confusion and concerns regarding hardware compatibility.

Adding the QuietBox 2 motherboard (`B850M-C`) and its standard PCIe bus mapping to `mobo_to_bus_ids` will silence this warning and provide official support for QuietBox 2 systems.

---

## 2. System & Environment Specifications

- **Workstation Model**: Tenstorrent QuietBox 2
- **Processors**: 4x Tenstorrent Blackhole accelerators (2x2 mesh topology, 440 Tensix cores)
- **Motherboard**: ASUS B850M-C (AM5 micro-ATX)
- **DMI Board Name**: `B850M-C`
  ```bash
  $ cat /sys/class/dmi/id/board_name
  B850M-C
  ```
- **PCI Bus Enumeration**:
  ```bash
  $ lspci -d 1e52:
  01:00.0 Processing accelerators: Tenstorrent Inc Blackhole
  02:00.0 Processing accelerators: Tenstorrent Inc Blackhole
  03:00.0 Processing accelerators: Tenstorrent Inc Blackhole
  04:00.0 Processing accelerators: Tenstorrent Inc Blackhole
  ```
- **Operating System**: Ubuntu 24.04.x LTS (Linux kernel 6.8+)
- **Software Stack**: TT-Metal 0.54+ / main, Tenstorrent KMD 2.11.0, firmware bundle 19.15.0

---

## 3. Root Cause

In `tt_metal/fabric/physical_system_discovery.cpp`:

1. `get_mobo_name()` reads `/sys/class/dmi/id/board_name`, returning `"B850M-C"`.
2. In `get_tray_id_for_chip(...)`:
   ```cpp
   static const std::unordered_map<std::string, std::vector<uint16_t>> mobo_to_bus_ids = {
       {"SIENAD8-2L2T", sienad8_canonical_bus_ids},
       {"X12DPG-QT6", {0xb1, 0xca, 0x31, 0x4b}},
       {"H13DSG-O-CPU", {0x21, 0x01, 0x41, 0x61, 0xa1, 0x81, 0xc1, 0xe1}},
   };
   ```
3. Because `"B850M-C"` is absent from the map, line 112 triggers:
   ```cpp
   if (!mobo_to_bus_ids.contains(mobo_name)) {
       auto bus_id = tt::tt_fabric::get_bus_id(cluster_desc, chip_id);
       static std::unordered_set<std::string> warned_mobo_names;
       if (warned_mobo_names.insert(mobo_name).second) {
           log_warning(
               tt::LogAlways,
               "Unknown motherboard '{}' for chip_id={} (bus_id=0x{:x}) — falling back to bus_id as tray_id. "
               "Add this motherboard and its bus IDs to mobo_to_bus_ids in physical_system_discovery.cpp.",
               mobo_name,
               chip_id,
               bus_id);
       }
       return TrayID{static_cast<uint32_t>(bus_id)};
   }
   ```

---

## 4. Proposed Fix (Patch)

Add the `"B850M-C"` entry to `mobo_to_bus_ids` in `tt_metal/fabric/physical_system_discovery.cpp`:

```diff
--- a/tt_metal/fabric/physical_system_discovery.cpp
+++ b/tt_metal/fabric/physical_system_discovery.cpp
@@ -91,6 +91,7 @@ TrayID get_tray_id_for_chip(
     static const std::unordered_map<std::string, std::vector<uint16_t>> mobo_to_bus_ids = {
         {"SIENAD8-2L2T", sienad8_canonical_bus_ids},
         {"X12DPG-QT6", {0xb1, 0xca, 0x31, 0x4b}},
         {"H13DSG-O-CPU", {0x21, 0x01, 0x41, 0x61, 0xa1, 0x81, 0xc1, 0xe1}},
+        {"B850M-C", {0x01, 0x02, 0x03, 0x04}},
     };
```

---

## 5. Verification & Expected Behavior

- **With Patch**:
  - `ordered_bus_ids` for `"B850M-C"` resolves to `{0x01, 0x02, 0x03, 0x04}`.
  - Tray IDs are computed as `std::distance(ordered_bus_ids.begin(), bus_id_it) + 1`:
    - Bus `0x01` $\to$ Tray 1
    - Bus `0x02` $\to$ Tray 2
    - Bus `0x03` $\to$ Tray 3
    - Bus `0x04` $\to$ Tray 4
  - No `LogAlways` warnings are emitted during device discovery.
