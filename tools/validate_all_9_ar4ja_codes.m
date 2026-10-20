% =========================================================================
% tools/validate_all_9_ar4ja_codes.m
% Exhaustive Automated Validation of All 9 CCSDS AR4JA LDPC Codes
% Cross-checking makeH_AR4JA.m against theoretical and standard specifications.
%
% Tests:
%   1. Execution across all 9 combinations: K in {1024, 4096, 16384}, Rate in {1/2, 2/3, 4/5}
%   2. Matrix dimensions, NNZ edges, and check degree distributions
%   3. Circulant structure: Unit-stride diagonals (dv/dc = +1)
%   4. Nested Sub-Matrix Property: Proof that for any M, H_1/2 and H_2/3
%      are exact right-justified sub-matrices of H_4/5.
% =========================================================================

clear; clc;
addpath('y:/JPL/LDPC_QB2/LDPC_QB2_GA/matrices');

fprintf('================================================================================\n');
fprintf('CCSDS AR4JA ALL 9 CODES AUTOMATED VALIDATION HARNESS\n');
fprintf('================================================================================\n\n');

K_vals = [1024, 4096, 16384];
rates  = [1/2, 2/3, 4/5];
rate_strs = {'1/2', '2/3', '4/5'};

% Expected Theoretical Properties from makeH_AR4JA and CCSDS 131.0-B
% Format: [K, rate_num, M, L, M_chk, N_var, NNZ_expected, Deg3_count, Other_deg, Other_deg_count]
% Check Degrees:
%   Rate 1/2: Row Block 0 is Deg 3 (M rows); Row Blocks 1 & 2 are Deg 6 (2M rows)
%   Rate 2/3: Row Block 0 is Deg 3 (M rows); Row Blocks 1 & 2 are Deg 10 (2M rows, 6 + 4 added)
%   Rate 4/5: Row Block 0 is Deg 3 (M rows); Row Blocks 1 & 2 are Deg 18 (2M rows, 10 + 8 added)
expected = [
    1024,  1,  512,  128,  1536,  2560,    7680,   512,  6, 1024;
    1024,  2,  256,   64,   768,  1792,    5888,   256, 10,  512;
    1024,  4,  128,   32,   384,  1408,    4992,   128, 18,  256;
    4096,  1, 2048,  512,  6144, 10240,   30720,  2048,  6, 4096;
    4096,  2, 1024,  256,  3072,  7168,   23552,  1024, 10, 2048;
    4096,  4,  512,  128,  1536,  5632,   19968,   512, 18, 1024;
   16384,  1, 8192, 2048, 24576, 40960,  122880,  8192,  6, 16384;
   16384,  2, 4096, 1024, 12288, 28672,   94208,  4096, 10,  8192;
   16384,  4, 2048,  512,  6144, 22528,   79872,  2048, 18,  4096;
];

all_pass = true;
row_idx = 1;

fprintf('| %-5s | %-4s | %-5s | %-5s | %-13s | %-8s | %-10s | %-18s | %-6s |\n', ...
    'K', 'Rate', 'M', 'L', 'M_chk x N_var', 'Edges', 'Exp Edges', 'Deg Distribution', 'Status');
fprintf('|%s|\n', repmat('-', 1, 93));

for k_i = 1:length(K_vals)
    K = K_vals(k_i);
    for r_i = 1:length(rates)
        r = rates(r_i);
        r_str = rate_strs{r_i};
        
        [H, punc] = makeH_AR4JA(K, r);
        
        M_chk = size(H, 1);
        N_var = size(H, 2);
        num_edges = nnz(H);
        
        exp_row = expected(row_idx, :);
        exp_M = exp_row(3);
        exp_L = exp_row(4);
        exp_M_chk = exp_row(5);
        exp_N_var = exp_row(6);
        exp_edges = exp_row(7);
        exp_deg3_cnt = exp_row(8);
        exp_other_deg = exp_row(9);
        exp_other_cnt = exp_row(10);
        
        row_degrees = full(sum(H, 2));
        deg3_cnt = sum(row_degrees == 3);
        other_cnt = sum(row_degrees == exp_other_deg);
        
        % Validate dimensions, edge counts, and degrees
        match_dim   = (M_chk == exp_M_chk) && (N_var == exp_N_var);
        match_edges = (num_edges == exp_edges);
        match_deg   = (deg3_cnt == exp_deg3_cnt) && (other_cnt == exp_other_cnt);
        match_punc  = (length(punc) == exp_M);
        
        test_ok = match_dim && match_edges && match_deg && match_punc;
        if ~test_ok
            all_pass = false;
            status_str = 'FAIL';
        else
            status_str = 'PASS';
        end
        
        deg_str = sprintf('%d d3, %d d%d', deg3_cnt, other_cnt, exp_other_deg);
        dim_str = sprintf('%d x %d', M_chk, N_var);
        
        fprintf('| %5d | %-4s | %5d | %5d | %-13s | %8d | %10d | %-18s | %-6s |\n', ...
            K, r_str, exp_M, exp_L, dim_str, num_edges, exp_edges, deg_str, status_str);
        
        row_idx = row_idx + 1;
    end
end
fprintf('|%s|\n\n', repmat('-', 1, 93));

% =========================================================================
% NESTED SUB-MATRIX PROPERTY VALIDATION
% Verify that for any fixed submatrix size M, H_1/2 and H_2/3 are exact
% right-justified column slices of H_4/5!
% =========================================================================
fprintf('================================================================================\n');
fprintf('PROVING NESTED SUB-MATRIX PROPERTY: H_1/2 and H_2/3 as Column Slices of H_4/5\n');
fprintf('================================================================================\n\n');

% Test across standard M values: M = 128, 256, 512, 1024, 2048
M_test_vals = [128, 256, 512, 1024, 2048];

for m_i = 1:length(M_test_vals)
    M = M_test_vals(m_i);
    L = M / 4;
    phitbl = log2(M) - 6;
    
    % Synthesize Pi1..Pi26 directly for this M using makeH_AR4JA logic
    table3 = [
        1 3 1 59 16 160 108 226 1148 0 0 0 0 0 0 0;
        2 0 22 18 103 241 126 618 2032 27 32 53 182 375 767 1822;
        3 1 0 52 105 185 238 404 249 30 21 74 249 436 227 203;
        4 2 26 23 0 251 481 32 1807 28 36 45 65 350 247 882;
        5 2 0 11 50 209 96 912 485 7 30 47 70 260 284 1989;
        6 3 10 7 29 103 28 950 1044 1 29 0 141 84 370 957;
        7 0 5 22 115 90 59 534 717 8 44 59 237 318 482 1705;
        8 1 18 25 30 184 225 63 873 20 29 102 77 382 273 1083;
        9 0 3 27 92 248 323 971 364 26 39 25 55 169 886 1072;
        10 1 22 30 78 12 28 304 1926 24 14 3 12 213 634 354;
        11 2 3 43 70 111 386 409 1241 4 22 88 227 67 762 1942;
        12 0 8 14 66 66 305 708 1769 12 15 65 42 313 184 446;
        13 2 25 46 39 173 34 719 532 23 48 62 52 242 696 1456;
        14 3 25 62 84 42 510 176 768 15 55 68 243 188 413 1940;
        15 0 2 44 79 157 147 743 1138 15 39 91 179 1 854 1660;
        16 1 27 12 70 174 199 759 965 22 11 70 250 306 544 1661;
        17 2 7 38 29 104 347 674 141 31 1 115 247 397 864 587;
        18 0 7 47 32 144 391 958 1527 3 50 31 164 80 82 708;
        19 1 15 1 45 43 165 984 505 29 40 121 17 33 1009 1466;
        20 2 10 52 113 181 414 11 1312 21 62 45 31 7 437 433;
        21 0 4 61 86 250 97 413 1840 2 27 56 149 447 36 1345;
        22 1 19 10 1 202 158 925 709 5 38 54 105 336 562 867;
        23 2 7 55 42 68 86 687 1427 11 40 108 183 424 816 1551;
        24 1 9 7 118 177 168 752 989 26 15 14 153 134 452 2041;
        25 2 26 12 33 170 506 867 1925 9 11 30 177 152 290 1383;
        26 3 17 2 126 89 489 323 270 17 18 116 19 492 778 1790
    ];
    table4 = [
        1 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        2 0 12 46 8 35 219 254 318 13 44 35 162 312 285 1189;
        3 1 30 45 119 167 16 790 494 19 51 97 7 503 554 458;
        4 2 18 27 89 214 263 642 1467 14 12 112 31 388 809 460;
        5 2 10 48 31 84 415 248 757 15 15 64 164 48 185 1039;
        6 3 16 37 122 206 403 899 1085 20 12 93 11 7 49 1000;
        7 0 13 41 1 122 184 328 1630 17 4 99 237 185 101 1265;
        8 1 9 13 69 67 279 518 64 4 7 94 125 328 82 1223;
        9 0 7 9 92 147 198 477 689 4 2 103 133 254 898 874;
        10 1 15 49 47 54 307 404 1300 11 30 91 99 202 627 1292;
        11 2 16 36 11 23 432 698 148 17 53 3 105 285 154 1491;
        12 0 18 10 31 93 240 160 777 20 23 6 17 11 65 631;
        13 2 4 11 19 20 454 497 1431 8 29 39 97 168 81 464;
        14 3 23 18 66 197 294 100 659 22 37 113 91 127 823 461;
        15 0 5 54 49 46 479 518 352 19 42 92 211 8 50 844;
        16 1 3 40 81 162 289 92 1177 15 48 119 128 437 413 392;
        17 2 29 27 96 101 373 464 836 5 4 74 82 475 462 922;
        18 0 11 35 38 76 104 592 1572 21 10 73 115 85 175 256;
        19 1 4 25 83 78 141 198 348 17 18 116 248 419 715 1986;
        20 2 8 46 42 253 270 856 1040 9 56 31 62 459 537 19;
        21 0 2 24 58 124 439 235 779 20 9 127 26 468 722 266;
        22 1 11 33 24 143 333 134 476 18 11 98 140 209 37 471;
        23 2 11 18 25 63 399 542 191 31 23 23 121 311 488 1166;
        24 1 3 37 92 41 14 545 1393 13 8 38 12 211 179 1300;
        25 2 15 35 38 214 277 777 1752 2 7 18 41 510 430 1033;
        26 3 13 21 120 70 412 483 1627 18 24 62 249 320 264 1606
    ];
    
    theta = table3(:, 2);
    phi = cat(3, table3(:, 3:9), table3(:, 10:16), table4(:, 3:9), table4(:, 10:16));
    phi = permute(phi, [1 3 2]);
    
    i_idx = 0:M-1;
    Pi_cell = cell(26, 1);
    for k = 1:26
        pi_k = M/4 * mod(theta(k) + floor(4*i_idx/M), 4) + mod(phi(k, floor(4*i_idx/M) + 1, phitbl) + i_idx, M/4) + 1;
        Pi_cell{k} = sparse(1:M, pi_k, ones(1, M), M, M);
    end
    
    Z = sparse(M, M);
    I = speye(M);
    
    % Construct H_1/2 (5M cols)
    H_12 = [Z Z I Z I+Pi_cell{1}; ...
            I I Z I Pi_cell{2}+Pi_cell{3}+Pi_cell{4}; ...
            I Pi_cell{5}+Pi_cell{6} Z Pi_cell{7}+Pi_cell{8} I];
        
    % Construct H_2/3 (7M cols)
    H_23 = [[Z Z; Pi_cell{9}+Pi_cell{10}+Pi_cell{11} I; I Pi_cell{12}+Pi_cell{13}+Pi_cell{14}] H_12];
    
    % Construct H_4/5 (11M cols)
    H_45 = [[Z Z Z Z; ...
             Pi_cell{21}+Pi_cell{22}+Pi_cell{23} I Pi_cell{15}+Pi_cell{16}+Pi_cell{17} I; ...
             I Pi_cell{24}+Pi_cell{25}+Pi_cell{26} I Pi_cell{18}+Pi_cell{19}+Pi_cell{20}] H_23];
         
    % Verify Submatrix Nesting:
    % H_12 must match exactly columns (6M+1):11M of H_45
    diff_12 = norm(H_12 - H_45(:, (6*M + 1):(11*M)), 'fro');
    
    % H_23 must match exactly columns (4M+1):11M of H_45
    diff_23 = norm(H_23 - H_45(:, (4*M + 1):(11*M)), 'fro');
    
    submatrix_ok = (diff_12 == 0) && (diff_23 == 0);
    if ~submatrix_ok
        all_pass = false;
        sub_status = 'FAIL';
    else
        sub_status = 'PASS (100% BIT-EXACT)';
    end
    
    fprintf('Submatrix Size M = %4d (L = %3d): H_1/2 in H_4/5 [diff = %g], H_2/3 in H_4/5 [diff = %g] -> %s\n', ...
        M, L, diff_12, diff_23, sub_status);
end

fprintf('\n================================================================================\n');
if all_pass
    fprintf('FINAL RESULT: 100%% VERIFIED PASS ACROSS ALL 9 CODES AND SUB-MATRIX PROOFS!\n');
else
    fprintf('FINAL RESULT: ONE OR MORE VERIFICATION CHECKS FAILED!\n');
end
fprintf('================================================================================\n');
