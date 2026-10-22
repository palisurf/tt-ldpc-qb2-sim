% =========================================================================
% PLOT_H_TILES_AND_FACES.M (Publication-Grade Infographic Edition v2)
% High-Impact Graphic for Wide Technical & Executive Audience:
% 1. Master Title Banner: CCSDS AR4JA Parity-Check Matrix Tiles & 512-bit DMA
% 2. Full Matrix H Spy Plot (40 Tiles, 160 Faces, Collision-Free Badges)
% 3. Detail Zooms: Degree-6 Tile 1 & Degree-3 Tile 33 (Zero-Collision Layout)
% 4. Section Divider Banner: Tensix Microarchitecture & 440-Core Throughput
% 5. Panel D: The Hardware-Code Match (8-CW SIMD, 16-Node Slice, 512-bit Bus)
% 6. Panel E: Tensix Core (Dual Couriers, Uniform TRISC Stages, Invariant Box)
% 7. Panel F: System Impact (Duty Cycle Gauges, 4-Chip Mesh, Spaced Badges)
%
% Author: Dr. Christopher R. Jones & Antigravity
% Target Hardware: Tenstorrent QuietBox 2 (4 Blackhole Mesh Device)
% =========================================================================

cd('y:/JPL/LDPC_QB2/LDPC_QB2_GA');
addpath('matrices');

fprintf('Loading CCSDS AR4JA Rate-1/2 matrix from makeH_AR4JA(1024, 0.5)...\n');
[H, punc] = makeH_AR4JA(1024, 0.5);
[M, N] = size(H);
fprintf('Loaded H: %d rows x %d cols, nnz = %d\n', M, N, nnz(H));

% -------------------------------------------------------------------------
% 1. Classify all 7,680 non-zeros into 40 Tiles and 160 Faces
% -------------------------------------------------------------------------
face_r = cell(4, 1);
face_c = cell(4, 1);
tile_bands = zeros(40, 4);

for t = 1:40
    if t <= 16
        r_start = 512 + (t - 1) * 32 + 1;
        r_end   = 512 + t * 32;
        deg = 6;
        sched_id = t;
    elseif t <= 32
        r_start = 1024 + (t - 17) * 32 + 1;
        r_end   = 1024 + (t - 16) * 32;
        deg = 6;
        sched_id = t;
    else
        r_start = (t - 33) * 64 + 1;
        r_end   = (t - 32) * 64;
        deg = 3;
        sched_id = t;
    end
    tile_bands(t, :) = [r_start, r_end, deg, sched_id];
    
    if deg == 6
        for r_local = 1:32
            r_global = r_start + r_local - 1;
            c_row = find(H(r_global, :));
            
            if r_local <= 16
                face_r{1} = [face_r{1}, repmat(r_global, 1, 3)];
                face_c{1} = [face_c{1}, c_row(1:3)];
                face_r{3} = [face_r{3}, repmat(r_global, 1, 3)];
                face_c{3} = [face_c{3}, c_row(4:6)];
            else
                face_r{2} = [face_r{2}, repmat(r_global, 1, 3)];
                face_c{2} = [face_c{2}, c_row(1:3)];
                face_r{4} = [face_r{4}, repmat(r_global, 1, 3)];
                face_c{4} = [face_c{4}, c_row(4:6)];
            end
        end
    else
        for r_local = 1:64
            r_global = r_start + r_local - 1;
            c_row = find(H(r_global, :));
            
            if r_local <= 16
                f_idx = 1;
            elseif r_local <= 32
                f_idx = 2;
            elseif r_local <= 48
                f_idx = 3;
            else
                f_idx = 4;
            end
            face_r{f_idx} = [face_r{f_idx}, repmat(r_global, 1, 3)];
            face_c{f_idx} = [face_c{f_idx}, c_row];
        end
    end
end

% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% 2. Setup Figure Canvas (2600 x 2550) with Refined Dark Theme
% -------------------------------------------------------------------------
fig_w = 2600;
fig_h = 2550;
fig = figure('Visible', 'off', 'Color', [0.030 0.040 0.070], 'Position', [10, 10, fig_w, fig_h]);
set(fig, 'InvertHardcopy', 'off');

col_f0 = [0.00, 0.85, 1.00]; % Cyan (Face 0)
col_f1 = [1.00, 0.60, 0.05]; % Orange (Face 1)
col_f2 = [0.88, 0.25, 1.00]; % Magenta (Face 2)
col_f3 = [0.00, 0.95, 0.50]; % Emerald Green (Face 3)
face_colors = {col_f0, col_f1, col_f2, col_f3};
face_names  = {'Face 0 (Worker 0: BRISC)', 'Face 1 (Worker 1: NCRISC)', ...
               'Face 2 (Worker 0: BRISC)', 'Face 3 (Worker 1: NCRISC)'};

% =========================================================================
% MASTER HEADER BANNER (y: 0.938 to 0.992, Expanded Width 0.950)
% =========================================================================
ax_hdr = axes('Position', [0.025, 0.938, 0.950, 0.054], 'Color', [0.06 0.08 0.14]);
axis(ax_hdr, 'off');
hold(ax_hdr, 'on');

rectangle(ax_hdr, 'Position', [0, 0, 1, 1], 'FaceColor', [0.07 0.10 0.18], ...
    'EdgeColor', [0.22 0.40 0.65], 'LineWidth', 1.5, 'Curvature', [0.08, 0.08]);

text(ax_hdr, 0.50, 0.70, 'Deep Space Network (DSN) CCSDS AR4JA Rate-1/2 LDPC: 4.91 Gsps Decoding', ...
    'Color', [1.0 1.0 1.0], 'FontSize', 30.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
text(ax_hdr, 0.50, 0.28, 'Circulant Structure Enables 512-bit Parallel Tensix L1 NOC DMA Gather/Scatter on QuietBox 2 (440 Cores)', ...
    'Color', [0.0 0.95 0.95], 'FontSize', 21.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');


% =========================================================================
% TOP SECTION: PARITY-CHECK MATRIX SPY PLOT & DETAIL ZOOMS (y: 0.540 to 0.955)
% =========================================================================

% PANEL A: Full Matrix Spy Plot (Left)
ax1 = axes('Position', [0.045, 0.550, 0.530, 0.355], 'Color', [0.02 0.025 0.045]);
hold(ax1, 'on');

for t = 1:40
    r1 = tile_bands(t, 1);
    r2 = tile_bands(t, 2);
    h = r2 - r1 + 1;
    if mod(t, 2) == 0
        rectangle(ax1, 'Position', [0.5, r1 - 0.5, 2560, h], ...
            'FaceColor', [0.055 0.065 0.10], 'EdgeColor', 'none');
    end
end

for bc = 512:512:2048
    line(ax1, [bc + 0.5, bc + 0.5], [0.5, 1536.5], 'Color', [0.22 0.28 0.38], 'LineWidth', 1.2, 'LineStyle', '--');
end
for br = 512:512:1024
    line(ax1, [0.5, 2560.5], [br + 0.5, br + 0.5], 'Color', [0.22 0.28 0.38], 'LineWidth', 1.5, 'LineStyle', '-');
end

% Plot fine circulant dots on the matrix canvas
for f = 1:4
    plot(ax1, face_c{f}, face_r{f}, '.', 'Color', face_colors{f}, 'MarkerSize', 3.6);
end

% Create dedicated large-dot proxy handles specifically for the legend
leg_handles = gobjects(4, 1);
for f = 1:4
    leg_handles(f) = plot(ax1, NaN, NaN, 'o', 'MarkerFaceColor', face_colors{f}, ...
        'MarkerEdgeColor', [0.90 0.95 1.00], 'LineWidth', 1.2, 'MarkerSize', 12, ...
        'DisplayName', face_names{f});
end

% Highlight Tile 1 (Deg 6) and Tile 33 (Deg 3)
rectangle(ax1, 'Position', [0.5, 512.5, 2560, 32], 'EdgeColor', [1.0 0.85 0.2], 'LineWidth', 2.2);
text(ax1, 2520, 528, 'Tile 1 Zoom (Deg 6) \rightarrow', 'Color', [1.0 0.85 0.2], 'FontSize', 16.0, ...
    'FontWeight', 'bold', 'HorizontalAlignment', 'right', 'BackgroundColor', [0.05 0.06 0.10], 'Margin', 4);

rectangle(ax1, 'Position', [0.5, 0.5, 2560, 64], 'EdgeColor', [0.2 0.85 1.0], 'LineWidth', 2.2);
text(ax1, 2520, 32, 'Tile 33 Zoom (Deg 3) \rightarrow', 'Color', [0.2 0.85 1.0], 'FontSize', 16.0, ...
    'FontWeight', 'bold', 'HorizontalAlignment', 'right', 'BackgroundColor', [0.05 0.06 0.10], 'Margin', 4);

% Row Block Labels placed in completely empty matrix regions (Zero data collision!)
% Row Block 0 (Rows 0..511): Cols 0..1024 are 100% EMPTY (Z matrices)
text(ax1, 50, 256, 'Row Block 0: 8 Deg-3 Check Tiles (Tiles 33..40, 64 checks/tile = 32 Faces)', ...
    'Color', [0.75 0.85 1.0], 'FontSize', 16.0, 'FontWeight', 'bold', 'BackgroundColor', [0.03 0.04 0.08], 'Margin', 5);

% Row Block 1 & 2: Cols 640..1536 are 100% EMPTY
text(ax1, 700, 768, 'Row Block 1: 16 Deg-6 Check Tiles (Tiles 1..16, 32 checks/tile = 64 Faces)', ...
    'Color', [0.75 0.85 1.0], 'FontSize', 16.0, 'FontWeight', 'bold', 'BackgroundColor', [0.03 0.04 0.08], 'Margin', 5);

text(ax1, 700, 1280, 'Row Block 2: 16 Deg-6 Check Tiles (Tiles 17..32, 32 checks/tile = 64 Faces)', ...
    'Color', [0.75 0.85 1.0], 'FontSize', 16.0, 'FontWeight', 'bold', 'BackgroundColor', [0.03 0.04 0.08], 'Margin', 5);

set(ax1, 'YDir', 'reverse', 'XLim', [0.5, 2560.5], 'YLim', [0.5, 1536.5], ...
    'XColor', [0.75 0.80 0.90], 'YColor', [0.75 0.80 0.90], 'FontSize', 14.0, 'Box', 'on', 'LineWidth', 1.2);
xlabel(ax1, 'Variable Node Columns v (0 ... 2559)   [5 Blocks of 512 Nodes: Deg 2, Deg 3, Deg 1, Deg 3, Deg 6 Punctured]', ...
    'FontSize', 16.5, 'FontWeight', 'bold', 'Color', [0.88 0.92 1.0]);
ylabel(ax1, 'Check Node Rows c (0 ... 1535)   [40 Tiles: 32 Deg-6 Tiles + 8 Deg-3 Tiles]', ...
    'FontSize', 16.5, 'FontWeight', 'bold', 'Color', [0.88 0.92 1.0]);
title(ax1, {'CCSDS AR4JA Parity-Check Matrix H (1536 \times 2560): 40 Tiles & 160 Circulant Faces', ...
            'Quasi-Cyclic Circulant Lift L = 128, Rate 1/2 (makeH\_AR4JA.m)'}, ...
      'FontSize', 20.0, 'FontWeight', 'bold', 'Color', [1 1 1]);

% Legend placed below Panel A with clean horizontal layout and prominent color dots
leg1 = legend(ax1, leg_handles, 'TextColor', [1 1 1], 'Color', [0.05 0.06 0.10], ...
       'EdgeColor', [0.26 0.31 0.42], 'FontSize', 15.0, 'Position', [0.045, 0.482, 0.530, 0.030], 'Orientation', 'horizontal');
try
    leg1.ItemTokenSize = [30, 24];
end


% PANEL B: Zoom on Degree-6 Tile (Tile 1: Checks 513..544, 32 checks, 4 Faces)
ax2 = axes('Position', [0.615, 0.745, 0.355, 0.160], 'Color', [0.02 0.025 0.045]);
hold(ax2, 'on');

t1_r_start = 513; t1_r_end = 544;
rectangle(ax2, 'Position', [0.5, t1_r_start - 0.5, 2560, 32], 'FaceColor', [0.055 0.065 0.10], 'EdgeColor', 'none');

for f = 1:4
    mask = (face_r{f} >= t1_r_start) & (face_r{f} <= t1_r_end);
    plot(ax2, face_c{f}(mask), face_r{f}(mask), 's', 'MarkerFaceColor', face_colors{f}, ...
        'MarkerEdgeColor', 'none', 'MarkerSize', 4.2);
    
    f_cols = face_c{f}(mask);
    f_rows = face_r{f}(mask);
    [sorted_cols, s_idx] = sort(f_cols);
    sorted_rows = f_rows(s_idx);
    
    split_pts = [1, find(diff(sorted_cols) > 40) + 1, length(sorted_cols) + 1];
    for sp = 1:length(split_pts)-1
        idx_r = split_pts(sp):(split_pts(sp+1)-1);
        sc_c = sorted_cols(idx_r);
        sc_r = sorted_rows(idx_r);
        min_c = min(sc_c); max_c = max(sc_c);
        min_r = min(sc_r); max_r = max(sc_r);
        rectangle(ax2, 'Position', [min_c - 1.5, min_r - 0.5, (max_c - min_c + 3.0), (max_r - min_r + 1.0)], ...
            'EdgeColor', face_colors{f}, 'LineWidth', 1.3, 'LineStyle', '-');
    end
end

for bc = 512:512:2048
    line(ax2, [bc + 0.5, bc + 0.5], [t1_r_start - 0.5, t1_r_end + 0.5], 'Color', [0.22 0.28 0.38], 'LineWidth', 0.9, 'LineStyle', '--');
end

% Labels placed in empty region (Cols 650..1450 are 100% empty) - Zero collision!
text(ax2, 650, 521, 'Face 0 & 2 (Worker 0: BRISC): Checks 513..528 [3 Slices \times 128 B = 384 B]', ...
    'Color', col_f0, 'FontSize', 15.0, 'FontWeight', 'bold', 'BackgroundColor', [0.04 0.05 0.09], 'Margin', 4);
text(ax2, 650, 537, 'Face 1 & 3 (Worker 1: NCRISC): Checks 529..544 [3 Slices \times 128 B = 384 B]', ...
    'Color', col_f1, 'FontSize', 15.0, 'FontWeight', 'bold', 'BackgroundColor', [0.04 0.05 0.09], 'Margin', 4);

set(ax2, 'YDir', 'reverse', 'XLim', [0.5, 2560.5], 'YLim', [t1_r_start - 0.5, t1_r_end + 0.5], ...
    'XColor', [0.75 0.80 0.90], 'YColor', [0.75 0.80 0.90], 'FontSize', 13.5, 'Box', 'on', 'LineWidth', 1.1);
ylabel(ax2, 'Checks 513..544', 'FontSize', 15.5, 'FontWeight', 'bold', 'Color', [0.85 0.90 0.98]);
title(ax2, {'DETAIL ZOOM: Degree-6 Tile 1 (32 Checks, 192 Edges)', ...
            'Face 0 & 2: Worker 0 (BRISC, Checks 0..15) | Face 1 & 3: Worker 1 (NCRISC, Checks 16..31)'}, ...
      'FontSize', 17.5, 'FontWeight', 'bold', 'Color', [1 0.85 0.2]);


% PANEL C: Zoom on Degree-3 Tile (Tile 33: Checks 1..64, 64 checks, 4 Faces)
ax3 = axes('Position', [0.615, 0.550, 0.355, 0.155], 'Color', [0.02 0.025 0.045]);
hold(ax3, 'on');

t33_r_start = 1; t33_r_end = 64;
rectangle(ax3, 'Position', [0.5, t33_r_start - 0.5, 2560, 64], 'FaceColor', [0.055 0.065 0.10], 'EdgeColor', 'none');

for f = 1:4
    mask = (face_r{f} >= t33_r_start) & (face_r{f} <= t33_r_end);
    plot(ax3, face_c{f}(mask), face_r{f}(mask), 's', 'MarkerFaceColor', face_colors{f}, ...
        'MarkerEdgeColor', 'none', 'MarkerSize', 4.2);
    
    f_cols = face_c{f}(mask);
    f_rows = face_r{f}(mask);
    [sorted_cols, s_idx] = sort(f_cols);
    sorted_rows = f_rows(s_idx);
    
    split_pts = [1, find(diff(sorted_cols) > 40) + 1, length(sorted_cols) + 1];
    for sp = 1:length(split_pts)-1
        idx_r = split_pts(sp):(split_pts(sp+1)-1);
        sc_c = sorted_cols(idx_r);
        sc_r = sorted_rows(idx_r);
        min_c = min(sc_c); max_c = max(sc_c);
        min_r = min(sc_r); max_r = max(sc_r);
        rectangle(ax3, 'Position', [min_c - 1.5, min_r - 0.5, (max_c - min_c + 3.0), (max_r - min_r + 1.0)], ...
            'EdgeColor', face_colors{f}, 'LineWidth', 1.3, 'LineStyle', '-');
    end
end

for bc = 512:512:2048
    line(ax3, [bc + 0.5, bc + 0.5], [t33_r_start - 0.5, t33_r_end + 0.5], 'Color', [0.22 0.28 0.38], 'LineWidth', 0.9, 'LineStyle', '--');
end

% Labels placed in empty region (Cols 0..950 are 100% empty) - Zero collision!
text(ax3, 50, 9,  'Face 0: Checks 1..16 (BRISC)', 'Color', col_f0, 'FontSize', 14.5, 'FontWeight', 'bold', 'BackgroundColor', [0.04 0.05 0.09], 'Margin', 3);
text(ax3, 50, 24, 'Face 1: Checks 17..32 (NCRISC)', 'Color', col_f1, 'FontSize', 14.5, 'FontWeight', 'bold', 'BackgroundColor', [0.04 0.05 0.09], 'Margin', 3);
text(ax3, 50, 39, 'Face 2: Checks 33..48 (BRISC)', 'Color', col_f2, 'FontSize', 14.5, 'FontWeight', 'bold', 'BackgroundColor', [0.04 0.05 0.09], 'Margin', 3);
text(ax3, 50, 54, 'Face 3: Checks 49..64 (NCRISC)', 'Color', col_f3, 'FontSize', 14.5, 'FontWeight', 'bold', 'BackgroundColor', [0.04 0.05 0.09], 'Margin', 3);

set(ax3, 'YDir', 'reverse', 'XLim', [0.5, 2560.5], 'YLim', [t33_r_start - 0.5, t33_r_end + 0.5], ...
    'XColor', [0.75 0.80 0.90], 'YColor', [0.75 0.80 0.90], 'FontSize', 13.5, 'Box', 'on', 'LineWidth', 1.1);
xlabel(ax3, 'Variable Node Columns v (0 ... 2559)', 'FontSize', 15.5, 'FontWeight', 'bold', 'Color', [0.85 0.90 0.98]);
ylabel(ax3, 'Checks 1..64', 'FontSize', 15.5, 'FontWeight', 'bold', 'Color', [0.85 0.90 0.98]);
title(ax3, {'DETAIL ZOOM: Degree-3 Tile 33 (64 Checks, 192 Edges)', ...
            'Face 0 (Checks 1..16), Face 1 (17..32), Face 2 (33..48), Face 3 (49..64)'}, ...
      'FontSize', 17.5, 'FontWeight', 'bold', 'Color', [0.2 0.85 1.0]);


% =========================================================================
% SECTION DIVIDER BANNER (y: 0.446 to 0.475)
% =========================================================================
ax_div = axes('Position', [0.045, 0.446, 0.925, 0.029], 'Color', [0.07 0.09 0.15]);
axis(ax_div, 'off');
hold(ax_div, 'on');

rectangle(ax_div, 'Position', [0, 0, 1, 1], 'FaceColor', [0.08 0.11 0.18], ...
    'EdgeColor', [0.22 0.40 0.65], 'LineWidth', 1.5, 'Curvature', [0.08, 0.08]);

text(ax_div, 0.5, 0.5, ...
    'TENSTORRENT QUIETBOX 2 TENSIX ARCHITECTURE: 38.016 TB/s via 512-BIT DMA DATA FLOW & 440-CORE THROUGHPUT', ...
    'Color', [0.0 0.95 0.95], 'FontSize', 25.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');


% =========================================================================
% BOTTOM SECTION: HIGHLY GRAPHICAL INFOGRAPHIC PANELS (y: 0.030 to 0.450)
% =========================================================================

% % -------------------------------------------------------------------------
% PANEL D: The Hardware-Code Match (Left Column: x: 0.035..0.335)
% -------------------------------------------------------------------------
ax_d = axes('Position', [0.035, 0.025, 0.300, 0.410], 'Color', [0.02 0.025 0.045]);
axis(ax_d, 'off');
hold(ax_d, 'on');

rectangle(ax_d, 'Position', [0, 0, 1, 1], 'FaceColor', [0.05 0.06 0.10], ...
    'EdgeColor', [0.18 0.28 0.44], 'LineWidth', 1.4, 'Curvature', [0.03, 0.03]);

text(ax_d, 0.05, 0.965, '1. THE HARDWARE-CODE MATCH', ...
    'Color', [0.0 0.85 1.0], 'FontSize', 19.0, 'FontWeight', 'bold');
line(ax_d, [0.05, 0.95], [0.938, 0.938], 'Color', [0.22 0.32 0.50], 'LineWidth', 1.2);

% --- Visual 1: 4-Codeword Packed Quartet (8 Bytes/Node) ---
text(ax_d, 0.05, 0.902, 'VARIABLE NODE MEMORY FORMAT (4 Codewords SIMD Quartet):', ...
    'Color', [1 1 1], 'FontSize', 16.5, 'FontWeight', 'bold');

cw_colors = [
    0.00, 0.85, 1.00;  % Cyan (CW0)
    1.00, 0.60, 0.05;  % Orange (CW1)
    0.88, 0.25, 1.00;  % Magenta (CW2)
    0.00, 0.95, 0.50   % Emerald Green (CW3)
];
w_cw = 0.210;
for cwi = 1:4
    x_cw = 0.05 + (cwi - 1) * 0.228;
    rectangle(ax_d, 'Position', [x_cw, 0.820, w_cw, 0.056], 'FaceColor', cw_colors(cwi, :), ...
        'EdgeColor', [1 1 1], 'LineWidth', 1.0, 'Curvature', [0.15, 0.15]);
    text(ax_d, x_cw + w_cw/2, 0.848, sprintf('CW%d', cwi-1), 'Color', [0 0 0], ...
        'FontSize', 17.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end
text(ax_d, 0.05, 0.776, '1 Variable Node = 4 \times bfloat16 = 8 BYTES TOTAL (64 BITS)', ...
    'Color', [0.85 0.90 1.0], 'FontSize', 15.5, 'FontWeight', 'bold');

% --- Visual 2: The 16-Variable Contiguous Circulant Slice ---
text(ax_d, 0.05, 0.720, 'CONTIGUOUS CIRCULANT SLICE (\Delta v = +1):', ...
    'Color', [1 1 1], 'FontSize', 16.5, 'FontWeight', 'bold');

v_box_x = [0.05, 0.18, 0.31, 0.44];
v_labels = {'v_0', 'v_1', 'v_2', 'v_3'};
for vi = 1:4
    rectangle(ax_d, 'Position', [v_box_x(vi), 0.638, 0.11, 0.055], 'FaceColor', [0.12 0.16 0.26], ...
        'EdgeColor', [0.0 0.85 1.0], 'LineWidth', 1.0, 'Curvature', [0.1, 0.1]);
    text(ax_d, v_box_x(vi) + 0.055, 0.665, v_labels{vi}, 'Color', [0.0 0.85 1.0], ...
        'FontSize', 16.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end
text(ax_d, 0.60, 0.665, '\bullet \bullet \bullet', 'Color', [0.7 0.8 0.9], 'FontSize', 18, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
rectangle(ax_d, 'Position', [0.68, 0.638, 0.11, 0.055], 'FaceColor', [0.12 0.16 0.26], ...
    'EdgeColor', [0.0 0.85 1.0], 'LineWidth', 1.0, 'Curvature', [0.1, 0.1]);
text(ax_d, 0.735, 0.665, 'v_{15}', 'Color', [0.0 0.85 1.0], ...
    'FontSize', 16.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

rectangle(ax_d, 'Position', [0.82, 0.630, 0.13, 0.068], 'FaceColor', [0.15 0.25 0.18], ...
    'EdgeColor', [0.0 0.95 0.5], 'LineWidth', 1.3, 'Curvature', [0.1, 0.1]);
text(ax_d, 0.885, 0.670, '128 B', 'Color', [0.0 0.95 0.5], 'FontSize', 17.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
text(ax_d, 0.885, 0.645, 'Slice', 'Color', [0.8 1 0.8], 'FontSize', 14.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

text(ax_d, 0.05, 0.598, '16 Nodes \times 8 Bytes = 128 BYTES CONTIGUOUS MEMORY', ...
    'Color', [0.0 0.95 0.5], 'FontSize', 15.5, 'FontWeight', 'bold');

% --- Visual 3: The 512-bit DMA Superhighway Conduit ---
rectangle(ax_d, 'Position', [0.05, 0.392, 0.90, 0.180], 'FaceColor', [0.08 0.14 0.18], ...
    'EdgeColor', [0.0 0.95 0.50], 'LineWidth', 1.8, 'Curvature', [0.05, 0.05]);

text(ax_d, 0.50, 0.535, 'TENSIX 512-BIT HARDWARE DMA BUS (64 BYTES / CYCLE)', ...
    'Color', [0.0 0.95 0.50], 'FontSize', 17.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

chev_x = [0.10, 0.22, 0.34, 0.46, 0.58, 0.70, 0.82];
for ci = 1:length(chev_x)
    text(ax_d, chev_x(ci), 0.475, '>>>>', 'Color', [0.0 0.85 0.4], 'FontSize', 22, 'FontWeight', 'bold');
end

rectangle(ax_d, 'Position', [0.06, 0.410, 0.88, 0.048], 'FaceColor', [0.04 0.08 0.10], ...
    'EdgeColor', [1.0 0.85 0.2], 'LineWidth', 1.2, 'Curvature', [0.08, 0.08]);
text(ax_d, 0.50, 0.434, 'BURST FORMULA:  128 Bytes \div 64 Bytes/cycle  =  2 CLOCK CYCLES!', ...
    'Color', [1.0 0.90 0.20], 'FontSize', 15.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

% --- Visual 4: Summary Card for Face & Tile Memory Bursts ---
rectangle(ax_d, 'Position', [0.05, 0.035, 0.90, 0.335], 'FaceColor', [0.07 0.09 0.15], ...
    'EdgeColor', [0.22 0.32 0.50], 'LineWidth', 1.4, 'Curvature', [0.04, 0.04]);

text(ax_d, 0.50, 0.340, 'CIRCULANT DMA TRANSFER SPECIFICATIONS', 'Color', [1 1 1], ...
    'FontSize', 18.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
line(ax_d, [0.08, 0.92], [0.320, 0.320], 'Color', [0.25 0.35 0.50], 'LineWidth', 1.0);

% Sub-Card 1: Per-Face Burst (Left side)
rectangle(ax_d, 'Position', [0.065, 0.048, 0.420, 0.258], 'FaceColor', [0.08 0.11 0.18], ...
    'EdgeColor', [0.00 0.85 1.00], 'LineWidth', 1.2, 'Curvature', [0.06, 0.06]);
rectangle(ax_d, 'Position', [0.065, 0.266, 0.420, 0.040], 'FaceColor', [0.10 0.16 0.28], ...
    'EdgeColor', [0.00 0.85 1.00], 'LineWidth', 0.8, 'Curvature', [0.08, 0.08]);
text(ax_d, 0.275, 0.286, 'PER-FACE BURST (16 Lanes)', 'Color', [0.00 0.85 1.00], ...
    'FontSize', 15.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

text(ax_d, 0.085, 0.232, 'Slices per Face:', 'Color', [0.85 0.90 1.0], 'FontSize', 14.0, 'FontWeight', 'bold');
text(ax_d, 0.085, 0.204, '3 Contiguous (48 Edges)', 'Color', [0.0 0.95 0.5], 'FontSize', 15.0, 'FontWeight', 'bold');

text(ax_d, 0.085, 0.160, 'Data Volume per Face:', 'Color', [0.85 0.90 1.0], 'FontSize', 14.0, 'FontWeight', 'bold');
text(ax_d, 0.085, 0.132, '3 \times 128 B = 384 Bytes', 'Color', [0.0 0.95 0.5], 'FontSize', 15.0, 'FontWeight', 'bold');

text(ax_d, 0.085, 0.088, 'DMA Burst Latency:', 'Color', [0.85 0.90 1.0], 'FontSize', 14.0, 'FontWeight', 'bold');
text(ax_d, 0.085, 0.060, '3 \times 2 cyc = 6 Clock Cycles', 'Color', [1.0 0.85 0.2], 'FontSize', 15.0, 'FontWeight', 'bold');

% Sub-Card 2: Per-Tile Concurrency (Right side)
rectangle(ax_d, 'Position', [0.515, 0.048, 0.420, 0.258], 'FaceColor', [0.07 0.12 0.14], ...
    'EdgeColor', [0.00 0.95 0.50], 'LineWidth', 1.2, 'Curvature', [0.06, 0.06]);
rectangle(ax_d, 'Position', [0.515, 0.266, 0.420, 0.040], 'FaceColor', [0.09 0.20 0.18], ...
    'EdgeColor', [0.00 0.95 0.50], 'LineWidth', 0.8, 'Curvature', [0.08, 0.08]);
text(ax_d, 0.725, 0.286, 'PER-TILE TOTALS (32 Lanes)', 'Color', [0.00 0.95 0.50], ...
    'FontSize', 15.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

text(ax_d, 0.535, 0.232, 'Total Slices per Tile:', 'Color', [0.85 0.90 1.0], 'FontSize', 14.0, 'FontWeight', 'bold');
text(ax_d, 0.535, 0.204, '6 Slices (768 Bytes)', 'Color', [0.0 0.95 0.5], 'FontSize', 15.0, 'FontWeight', 'bold');

text(ax_d, 0.535, 0.160, 'Parallel DMA Couriers:', 'Color', [0.85 0.90 1.0], 'FontSize', 14.0, 'FontWeight', 'bold');
text(ax_d, 0.535, 0.132, '2 Workers (BRISC & NCRISC)', 'Color', [0.0 0.95 0.5], 'FontSize', 15.0, 'FontWeight', 'bold');

text(ax_d, 0.535, 0.088, 'Total Gather & Scatter:', 'Color', [0.85 0.90 1.0], 'FontSize', 14.0, 'FontWeight', 'bold');
text(ax_d, 0.535, 0.060, '12 Clock Cycles (Overlapped)', 'Color', [1.0 0.85 0.2], 'FontSize', 15.0, 'FontWeight', 'bold');


% -------------------------------------------------------------------------
% PANEL E: Tensix Core Execution Engine (Middle Column: x: 0.360..0.695)
% -------------------------------------------------------------------------
ax_e = axes('Position', [0.360, 0.025, 0.335, 0.410], 'Color', [0.02 0.025 0.045]);
axis(ax_e, 'off');
hold(ax_e, 'on');

rectangle(ax_e, 'Position', [0, 0, 1, 1], 'FaceColor', [0.05 0.06 0.10], ...
    'EdgeColor', [0.18 0.28 0.44], 'LineWidth', 1.4, 'Curvature', [0.03, 0.03]);

text(ax_e, 0.05, 0.965, '2. TENSIX CORE: 5 WORKERS IN HARMONY', ...
    'Color', [1.0 0.85 0.2], 'FontSize', 19.0, 'FontWeight', 'bold');
line(ax_e, [0.05, 0.95], [0.938, 0.938], 'Color', [0.22 0.32 0.50], 'LineWidth', 1.2);

% --- Data Movement Plane: BRISC & NCRISC ---
text(ax_e, 0.05, 0.902, 'DATA MOVEMENT PLANE (Dual Dedicated DMA Couriers):', ...
    'Color', [1 1 1], 'FontSize', 16.5, 'FontWeight', 'bold');

% Worker 0 (BRISC) Card
rectangle(ax_e, 'Position', [0.045, 0.740, 0.440, 0.145], 'FaceColor', [0.07 0.11 0.18], ...
    'EdgeColor', col_f0, 'LineWidth', 1.4, 'Curvature', [0.06, 0.06]);
text(ax_e, 0.070, 0.845, 'WORKER 0: BRISC', 'Color', col_f0, 'FontSize', 18.0, 'FontWeight', 'bold');
text(ax_e, 0.070, 0.795, '\bullet Streams Face 0 & Face 2 (16 Lanes)', 'Color', [0.85 0.90 0.98], 'FontSize', 14.5, 'FontWeight', 'bold');
text(ax_e, 0.070, 0.755, '\bullet 3 Slices \times 128 B = 384 B (6 cyc)', 'Color', [0.0 0.95 0.5], 'FontSize', 14.5, 'FontWeight', 'bold');

% Worker 1 (NCRISC) Card
rectangle(ax_e, 'Position', [0.515, 0.740, 0.440, 0.145], 'FaceColor', [0.07 0.11 0.18], ...
    'EdgeColor', col_f1, 'LineWidth', 1.4, 'Curvature', [0.06, 0.06]);
text(ax_e, 0.540, 0.845, 'WORKER 1: NCRISC', 'Color', col_f1, 'FontSize', 18.0, 'FontWeight', 'bold');
text(ax_e, 0.540, 0.795, '\bullet Streams Face 1 & Face 3 (16 Lanes)', 'Color', [0.85 0.90 0.98], 'FontSize', 14.5, 'FontWeight', 'bold');
text(ax_e, 0.540, 0.755, '\bullet 3 Slices \times 128 B = 384 B (6 cyc)', 'Color', [0.0 0.95 0.5], 'FontSize', 14.5, 'FontWeight', 'bold');

% --- Compute Engine: TRISC Tri-Core & 32-Lane SFPU ---
text(ax_e, 0.05, 0.680, 'COMPUTE ENGINE (32-Lane SIMD Vector SFPU Turbine):', ...
    'Color', [1 1 1], 'FontSize', 16.5, 'FontWeight', 'bold');

rectangle(ax_e, 'Position', [0.045, 0.280, 0.910, 0.380], 'FaceColor', [0.07 0.09 0.16], ...
    'EdgeColor', [0.0 0.95 0.5], 'LineWidth', 1.5, 'Curvature', [0.04, 0.04]);

% 3 Sub-stages: UNPACK -> MATH SFPU -> PACK
stage_y = 0.490; stage_h = 0.145;

% TRISC0: UNPACK
rectangle(ax_e, 'Position', [0.065, stage_y, 0.255, stage_h], 'FaceColor', [0.10 0.13 0.22], ...
    'EdgeColor', [0.4 0.5 0.7], 'LineWidth', 1.0, 'Curvature', [0.06, 0.06]);
text(ax_e, 0.192, stage_y + 0.098, 'TRISC0: UNPACK', 'Color', [0.8 0.9 1.0], 'FontSize', 16.5, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(ax_e, 0.192, stage_y + 0.045, 'L1 \rightarrow DEST Regs', 'Color', [0.65 0.75 0.90], 'FontSize', 14.5, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

text(ax_e, 0.335, stage_y + 0.070, '\rightarrow', 'Color', [0.0 0.95 0.5], 'FontSize', 22, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

% TRISC1: MATH SFPU
rectangle(ax_e, 'Position', [0.355, stage_y, 0.290, stage_h], 'FaceColor', [0.12 0.18 0.28], ...
    'EdgeColor', [0.0 0.95 0.5], 'LineWidth', 1.4, 'Curvature', [0.06, 0.06]);
text(ax_e, 0.500, stage_y + 0.105, 'TRISC1: MATH SFPU', 'Color', [0.0 0.95 0.5], 'FontSize', 17.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(ax_e, 0.500, stage_y + 0.065, '32-Lane SIMD Vector', 'Color', [1 1 1], 'FontSize', 14.5, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(ax_e, 0.500, stage_y + 0.025, 'Approx-Min* Kernel', 'Color', [0.8 0.9 1.0], 'FontSize', 14.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

text(ax_e, 0.665, stage_y + 0.070, '\rightarrow', 'Color', [0.0 0.95 0.5], 'FontSize', 22, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

% TRISC2: PACK
rectangle(ax_e, 'Position', [0.685, stage_y, 0.255, stage_h], 'FaceColor', [0.10 0.13 0.22], ...
    'EdgeColor', [0.4 0.5 0.7], 'LineWidth', 1.0, 'Curvature', [0.06, 0.06]);
text(ax_e, 0.812, stage_y + 0.098, 'TRISC2: PACK', 'Color', [0.8 0.9 1.0], 'FontSize', 16.5, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(ax_e, 0.812, stage_y + 0.045, 'DEST \rightarrow L1 Scratch', 'Color', [0.65 0.75 0.90], 'FontSize', 14.5, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% DEST Registers box
rectangle(ax_e, 'Position', [0.065, 0.370, 0.870, 0.095], 'FaceColor', [0.05 0.07 0.12], ...
    'EdgeColor', [0.3 0.4 0.6], 'LineWidth', 1.0, 'Curvature', [0.05, 0.05]);
text(ax_e, 0.50, 0.430, '64 DEST REGISTERS (Native 16-bit Mode: dst\_reg[0..63])', ...
    'Color', [1 1 1], 'FontSize', 15.5, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(ax_e, 0.50, 0.392, 'Tile 0: dst\_reg[0..31] (R\_new Check)  |  Tile 1: dst\_reg[32..63] (L\_new Post LLR)', ...
    'Color', [0.0 0.85 1.0], 'FontSize', 13.5, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Active Math Time Box
rectangle(ax_e, 'Position', [0.065, 0.295, 0.870, 0.058], 'FaceColor', [0.08 0.14 0.12], ...
    'EdgeColor', [0.0 0.95 0.5], 'LineWidth', 1.0, 'Curvature', [0.05, 0.05]);
text(ax_e, 0.50, 0.324, 'ACTIVE MATH TIME: 912 CYCLES / TILE (All 4 Faces in Parallel)', ...
    'Color', [0.0 0.95 0.5], 'FontSize', 16.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

% --- Hardware Breakthrough Box: Zero-Transpose SIMD & Lockstep DMA ---
rectangle(ax_e, 'Position', [0.045, 0.035, 0.910, 0.230], 'FaceColor', [0.07 0.08 0.13], ...
    'EdgeColor', [0.00 0.85 1.00], 'LineWidth', 1.4, 'Curvature', [0.04, 0.04]);

% Title Banner across top
rectangle(ax_e, 'Position', [0.045, 0.222, 0.910, 0.043], 'FaceColor', [0.06 0.15 0.24], ...
    'EdgeColor', [0.00 0.85 1.00], 'LineWidth', 1.0, 'Curvature', [0.08, 0.08]);
text(ax_e, 0.50, 0.243, 'HARDWARE INVARIANTS: Zero-Transpose SIMD & Lockstep DMA', ...
    'Color', [0.00 0.95 0.95], 'FontSize', 17.5, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

% Sub-Card 1: Zero-Transpose Vertical SIMD Math (Left side)
rectangle(ax_e, 'Position', [0.060, 0.048, 0.430, 0.162], 'FaceColor', [0.06 0.12 0.18], ...
    'EdgeColor', [0.00 0.85 1.00], 'LineWidth', 1.2, 'Curvature', [0.06, 0.06]);
rectangle(ax_e, 'Position', [0.060, 0.172, 0.430, 0.038], 'FaceColor', [0.08 0.18 0.26], ...
    'EdgeColor', [0.00 0.85 1.00], 'LineWidth', 0.8, 'Curvature', [0.08, 0.08]);
text(ax_e, 0.275, 0.191, 'ZERO-TRANSPOSE VERTICAL SIMD', 'Color', [0.00 0.85 1.00], ...
    'FontSize', 14.5, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

text(ax_e, 0.075, 0.142, '\bullet Edges 0..2 (Face 0) & 3..5 (Face 2)', 'Color', [0.92 0.94 1.0], 'FontSize', 14.0, 'FontWeight', 'bold');
text(ax_e, 0.075, 0.102, '\bullet Pure vertical column vector math', 'Color', [0.92 0.94 1.0], 'FontSize', 14.0, 'FontWeight', 'bold');
text(ax_e, 0.075, 0.062, '\bullet 100% of SFPTRANSP eliminated!', 'Color', [0.0 0.95 0.5], 'FontSize', 14.0, 'FontWeight', 'bold');

% Sub-Card 2: Strict Lockstep & Convergence Latching (Right side)
rectangle(ax_e, 'Position', [0.510, 0.048, 0.430, 0.162], 'FaceColor', [0.06 0.14 0.10], ...
    'EdgeColor', [0.00 0.95 0.50], 'LineWidth', 1.2, 'Curvature', [0.06, 0.06]);
rectangle(ax_e, 'Position', [0.510, 0.172, 0.430, 0.038], 'FaceColor', [0.08 0.22 0.14], ...
    'EdgeColor', [0.00 0.95 0.50], 'LineWidth', 0.8, 'Curvature', [0.08, 0.08]);
text(ax_e, 0.725, 0.191, 'LOCKSTEP & CONVERGENCE LATCH', 'Color', [0.00 0.95 0.50], ...
    'FontSize', 14.5, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

text(ax_e, 0.525, 0.142, '\bullet Sequential Gather \rightarrow Math \rightarrow Scatter', 'Color', [0.92 0.94 1.0], 'FontSize', 14.0, 'FontWeight', 'bold');
text(ax_e, 0.525, 0.102, '\bullet Blind DMA scatter across 4 CW', 'Color', [0.92 0.94 1.0], 'FontSize', 14.0, 'FontWeight', 'bold');
text(ax_e, 0.525, 0.062, '\bullet Zero RAW hazards; 94.2% Duty Cycle!', 'Color', [0.0 0.95 0.5], 'FontSize', 14.0, 'FontWeight', 'bold');


% -------------------------------------------------------------------------
% PANEL F: System Performance & Throughput (Right Column: x: 0.710..0.970)
% -------------------------------------------------------------------------
ax_f = axes('Position', [0.710, 0.025, 0.260, 0.410], 'Color', [0.02 0.025 0.045]);
axis(ax_f, 'off');
hold(ax_f, 'on');

rectangle(ax_f, 'Position', [0, 0, 1, 1], 'FaceColor', [0.05 0.06 0.10], ...
    'EdgeColor', [0.18 0.28 0.44], 'LineWidth', 1.4, 'Curvature', [0.03, 0.03]);

text(ax_f, 0.05, 0.965, '3. SYSTEM IMPACT: 440 CORES', ...
    'Color', [0.0 0.95 0.5], 'FontSize', 21.0, 'FontWeight', 'bold');
line(ax_f, [0.05, 0.95], [0.938, 0.938], 'Color', [0.22 0.32 0.50], 'LineWidth', 1.2);

% --- Visual Efficiency Gauges: Before vs After ---
text(ax_f, 0.05, 0.900, 'COMPUTE DUTY CYCLE COMPARISON:', 'Color', [1 1 1], 'FontSize', 18.0, 'FontWeight', 'bold');

text(ax_f, 0.05, 0.850, 'Baseline (Scalar Ptrs): 3,400 cyc', 'Color', [0.8 0.85 0.95], 'FontSize', 17.0, 'FontWeight', 'bold');
rectangle(ax_f, 'Position', [0.05, 0.792, 0.90, 0.044], 'FaceColor', [0.45 0.15 0.15], 'EdgeColor', 'none', 'Curvature', [0.2, 0.2]);
rectangle(ax_f, 'Position', [0.05, 0.792, 0.90 * 0.268, 0.044], 'FaceColor', [0.00 0.75 0.40], 'EdgeColor', 'none', 'Curvature', [0.2, 0.2]);
text(ax_f, 0.17, 0.814, '26.8% Compute', 'Color', [1 1 1], 'FontSize', 16.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
text(ax_f, 0.62, 0.814, '73.2% MEMORY STALL', 'Color', [1.0 0.7 0.7], 'FontSize', 16.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

text(ax_f, 0.05, 0.748, 'Circulant 512-bit DMA: 968 cyc', 'Color', [0.0 0.95 0.5], 'FontSize', 17.5, 'FontWeight', 'bold');
rectangle(ax_f, 'Position', [0.05, 0.690, 0.90, 0.044], 'FaceColor', [0.20 0.25 0.35], 'EdgeColor', 'none', 'Curvature', [0.2, 0.2]);
rectangle(ax_f, 'Position', [0.05, 0.690, 0.90 * 0.942, 0.044], 'FaceColor', [0.00 0.95 0.50], 'EdgeColor', 'none', 'Curvature', [0.2, 0.2]);
text(ax_f, 0.50, 0.712, '94.2% ACTIVE SFPU COMPUTE!', 'Color', [0 0 0], 'FontSize', 16.5, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

text(ax_f, 0.05, 0.655, '\Delta 12-cycle DMA 100% submerged in lockstep!', 'Color', [0.0 0.95 0.5], 'FontSize', 17.0, 'FontWeight', 'bold');

line(ax_f, [0.05, 0.95], [0.630, 0.630], 'Color', [0.22 0.32 0.50], 'LineWidth', 1.0);

% --- Visual 440-Core QuietBox 2 Mesh Topology Box ---
text(ax_f, 0.05, 0.595, 'QUIETBOX 2 MESH (4 BLACKHOLE CHIPS):', 'Color', [1.0 0.85 0.2], 'FontSize', 18.0, 'FontWeight', 'bold');

chip_w = 0.42; chip_h = 0.060;
chip_coords = [
    0.05, 0.510;
    0.53, 0.510;
    0.05, 0.430;
    0.53, 0.430
];
chip_names = {'CHIP 0 (110 Cores)', 'CHIP 1 (110 Cores)', 'CHIP 2 (110 Cores)', 'CHIP 3 (110 Cores)'};

for ch = 1:4
    rectangle(ax_f, 'Position', [chip_coords(ch, 1), chip_coords(ch, 2), chip_w, chip_h], ...
        'FaceColor', [0.08 0.12 0.20], 'EdgeColor', [0.0 0.85 1.0], 'LineWidth', 1.0, 'Curvature', [0.1, 0.1]);
    text(ax_f, chip_coords(ch, 1) + chip_w/2, chip_coords(ch, 2) + chip_h/2, chip_names{ch}, ...
        'Color', [0.85 0.92 1.0], 'FontSize', 17.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end

% Interconnect lines
line(ax_f, [0.47, 0.53], [0.540, 0.540], 'Color', [0.0 0.95 0.5], 'LineWidth', 2.0);
line(ax_f, [0.47, 0.53], [0.460, 0.460], 'Color', [0.0 0.95 0.5], 'LineWidth', 2.0);
line(ax_f, [0.26, 0.26], [0.490, 0.510], 'Color', [0.0 0.95 0.5], 'LineWidth', 2.0);
line(ax_f, [0.74, 0.74], [0.490, 0.510], 'Color', [0.0 0.95 0.5], 'LineWidth', 2.0);

rectangle(ax_f, 'Position', [0.05, 0.365, 0.90, 0.046], 'FaceColor', [0.10 0.16 0.25], ...
    'EdgeColor', [1.0 0.85 0.2], 'LineWidth', 1.2, 'Curvature', [0.1, 0.1]);
text(ax_f, 0.50, 0.388, 'TOTAL: 440 TENSIX CORES IN 2x2 MESH', ...
    'Color', [1.0 0.85 0.2], 'FontSize', 18.5, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

% --- Big Impact Performance Badges ---
rectangle(ax_f, 'Position', [0.05, 0.210, 0.90, 0.140], 'FaceColor', [0.08 0.16 0.14], ...
    'EdgeColor', [0.0 0.95 0.5], 'LineWidth', 1.8, 'Curvature', [0.06, 0.06]);
text(ax_f, 0.50, 0.316, 'SUSTAINED MESH THROUGHPUT', 'Color', [0.8 1.0 0.9], 'FontSize', 17.5, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(ax_f, 0.50, 0.268, '4.91 Gsps', 'Color', [0.0 0.95 0.5], 'FontSize', 38.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(ax_f, 0.50, 0.230, '2.46 Gbps Information Bit Rate (Rate-1/2)', 'Color', [0.85 0.92 1.0], 'FontSize', 16.5, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

rectangle(ax_f, 'Position', [0.05, 0.045, 0.90, 0.145], 'FaceColor', [0.15 0.12 0.24], ...
    'EdgeColor', [0.85 0.25 1.0], 'LineWidth', 1.8, 'Curvature', [0.06, 0.06]);
text(ax_f, 0.50, 0.156, 'NET HARDWARE ACCELERATION', 'Color', [0.9 0.8 1.0], 'FontSize', 17.5, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(ax_f, 0.50, 0.108, '29.5x MULTIPLIER', 'Color', [0.90 0.35 1.0], 'FontSize', 38.0, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(ax_f, 0.50, 0.068, 'vs Verified 5-Worker Scalar Baseline (166 Msps)', 'Color', [0.85 0.90 0.98], 'FontSize', 15.5, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% -------------------------------------------------------------------------
% 3. Export Polished Figure (Widescreen Unconstrained Canvas - No Letter Paper Constraint!)
% -------------------------------------------------------------------------
out_png = 'y:/JPL/LDPC_QB2/LDPC_QB2_GA/docs/figures/ar4ja_h_matrix_tiles_and_faces_spy_plot.png';
out_pdf = 'y:/JPL/LDPC_QB2/LDPC_QB2_GA/docs/figures/ar4ja_h_matrix_tiles_and_faces_spy_plot.pdf';

% Set explicit Paper dimensions matching canvas width and height (NO 8.5 x 11 squashing!)
set(fig, 'PaperUnits', 'points');
set(fig, 'PaperSize', [fig_w, fig_h]);
set(fig, 'PaperPosition', [0, 0, fig_w, fig_h]);
set(fig, 'PaperPositionMode', 'manual');
set(fig, 'InvertHardcopy', 'off');

fprintf('Exporting high-resolution PNG (300 DPI) to %s...\n', out_png);
exportgraphics(fig, out_png, 'Resolution', 300, 'BackgroundColor', 'current');

fprintf('Exporting unconstrained vector PDF to %s...\n', out_pdf);
print(fig, out_pdf, '-dpdf', '-painters');

close(fig);
fprintf('SUCCESS: Generated master consumable infographic v2!\n');
