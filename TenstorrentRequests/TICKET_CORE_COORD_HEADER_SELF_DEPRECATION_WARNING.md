# Bug Report: Self-Referential Deprecation Warning in `core_coord.hpp` Header Wrappers

- **Component**: TT-Metalium Core Coordinate & Topology Headers
- **Header File**: `tt_metal/include/tt-metalium/core_coord.hpp` (also reflected in `build/include/tt-metalium/core_coord.hpp`)
- **Target Hardware**: Tenstorrent QuietBox 2 / Blackhole / Wormhole
- **Severity**: Low / Build Cleanliness (Spurious compiler warning emitted during downstream C++ compilation)
- **Reported Date**: 2026-09-28

---

## 1. Summary

When compiling any user application with `-Wall -Wextra` that includes standard TT-Metalium headers (such as `<tt-metalium/host_api.hpp>` or `<tt-metalium/core_coord.hpp>`), the GCC/Clang compiler emits a deprecation warning originated entirely within `core_coord.hpp`:

```text
In file included from /home/ttuser/tt-metal/build/include/tt-metalium/buffer_distribution_spec.hpp:8,
                 from /home/ttuser/tt-metal/build/include/tt-metalium/buffer.hpp:18,
                 from /home/ttuser/tt-metal/build/include/tt-metalium/host_api.hpp:12,
                 from ldpc_sim.cpp:11:
/home/ttuser/tt-metal/build/include/tt-metalium/core_coord.hpp:250:100: warning: ‘using CoreRange = class tt::tt_metal::CoreRange’ is deprecated: Use tt::tt_metal::CoreRange [-Wdeprecated-declarations]
  250 | [[deprecated("Use tt::tt_metal::select_contiguous_range_from_corerangeset")]] inline std::optional<CoreRange>
      |                                                                                                    ^~~~~~~~~
/home/ttuser/tt-metal/build/include/tt-metalium/core_coord.hpp:213:7: note: declared here
  213 | using CoreRange [[deprecated("Use tt::tt_metal::CoreRange")]] = tt::tt_metal::CoreRange;
      |       ^~~~~~~~~
```

This warning occurs even if the downstream user code **never uses** the deprecated `CoreRange` alias or the deprecated wrapper functions.

---

## 2. Root Cause Analysis

In `tt_metal/include/tt-metalium/core_coord.hpp`:

1. **Lines 213–214** introduce the deprecation alias to transition types into the `tt::tt_metal` namespace:
   ```cpp
   // core_coord.hpp:213-214
   using CoreRange [[deprecated("Use tt::tt_metal::CoreRange")]] = tt::tt_metal::CoreRange;
   using CoreRangeSet [[deprecated("Use tt::tt_metal::CoreRangeSet")]] = tt::tt_metal::CoreRangeSet;
   ```

2. **Lines 249–253** define deprecated backward-compatibility wrapper functions in the outer namespace. However, the signature of `select_contiguous_range_from_corerangeset` references the unqualified, now-deprecated `CoreRange` alias as its return type (`std::optional<CoreRange>`):
   ```cpp
   // core_coord.hpp:249-253
   template <bool _compiler_deprioritize_this = true>
   [[deprecated("Use tt::tt_metal::select_contiguous_range_from_corerangeset")]] inline std::optional<CoreRange>
   select_contiguous_range_from_corerangeset(const CoreRangeSet& crs, uint32_t x, uint32_t y) {
       return tt::tt_metal::select_contiguous_range_from_corerangeset(crs, x, y);
   }
   ```

3. When a downstream C++ file includes `<tt-metalium/host_api.hpp>`, the compiler parses the template function declaration and evaluates the return type `std::optional<CoreRange>`. Because `CoreRange` is tagged with `[[deprecated]]` on line 213, the compiler emits a `-Wdeprecated-declarations` warning on the header itself.

Similarly, other wrapper declarations in the same block (`corerange_to_cores`, `grid_to_cores_with_noop`) use unqualified `const CoreRangeSet&` in their parameter lists, which can trigger additional deprecation warnings under strict warning flags.

---

## 3. Impact on Users

- Downstream applications cannot compile with `-Werror` (warnings-as-errors) without suppressing `-Wno-deprecated-declarations` globally.
- Emits confusing warnings that suggest user code is using deprecated APIs when the deprecation is strictly internal to TT-Metalium headers.

---

## 4. Proposed Solution

Qualify the return type and parameter types in the deprecated wrapper signatures with `tt::tt_metal::`:

```diff
--- a/tt_metal/include/tt-metalium/core_coord.hpp
+++ b/tt_metal/include/tt-metalium/core_coord.hpp
@@ -218,7 +218,7 @@ using CoreRangeSet [[deprecated("Use tt::tt_metal::CoreRangeSet")]] = tt::tt_me
 
 template <bool _compiler_deprioritize_this = true>
 [[deprecated("Use tt::tt_metal::corerange_to_cores")]] inline std::vector<tt::tt_metal::CoreCoord> corerange_to_cores(
-    const CoreRangeSet& crs, std::optional<uint32_t> max_cores = std::nullopt, bool row_wise = false) {
+    const tt::tt_metal::CoreRangeSet& crs, std::optional<uint32_t> max_cores = std::nullopt, bool row_wise = false) {
     return tt::tt_metal::corerange_to_cores(crs, max_cores, row_wise);
 }
 
@@ -244,12 +244,12 @@ template <bool _compiler_deprioritize_this = true>
 
 template <bool _compiler_deprioritize_this = true>
 [[deprecated("Use tt::tt_metal::grid_to_cores_with_noop")]] inline std::vector<tt::tt_metal::CoreCoord> grid_to_cores_with_noop(
-    const CoreRangeSet& used_cores, const CoreRangeSet& all_cores, bool row_wise = false) {
+    const tt::tt_metal::CoreRangeSet& used_cores, const tt::tt_metal::CoreRangeSet& all_cores, bool row_wise = false) {
     return tt::tt_metal::grid_to_cores_with_noop(used_cores, all_cores, row_wise);
 }
 
 template <bool _compiler_deprioritize_this = true>
-[[deprecated("Use tt::tt_metal::select_contiguous_range_from_corerangeset")]] inline std::optional<CoreRange>
-select_contiguous_range_from_corerangeset(const CoreRangeSet& crs, uint32_t x, uint32_t y) {
+[[deprecated("Use tt::tt_metal::select_contiguous_range_from_corerangeset")]] inline std::optional<tt::tt_metal::CoreRange>
+select_contiguous_range_from_corerangeset(const tt::tt_metal::CoreRangeSet& crs, uint32_t x, uint32_t y) {
     return tt::tt_metal::select_contiguous_range_from_corerangeset(crs, x, y);
 }
```

Alternatively, `#pragma GCC diagnostic push` / `#pragma GCC diagnostic ignored "-Wdeprecated-declarations"` can wrap lines 215–260 of `core_coord.hpp` to ensure the backward-compatibility wrappers do not warn upon header inclusion.
