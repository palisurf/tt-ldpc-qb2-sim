import numpy as np

def parse_makeH_tables(matlab_file):
    with open(matlab_file, 'r') as f:
        lines = f.readlines()
    
    table2_lines = []
    table3_lines = []
    table4_lines = []
    
    current_table = None
    for line in lines:
        line_s = line.strip()
        if 'table2=[' in line_s:
            current_table = 2
            continue
        elif 'table3=[' in line_s:
            current_table = 3
            continue
        elif 'table4=[' in line_s:
            current_table = 4
            continue
        
        if current_table == 2:
            cleaned = line_s.replace('];', '').strip()
            if cleaned:
                table2_lines.append([int(x) for x in cleaned.split()])
            if '];' in line_s:
                current_table = None
        elif current_table == 3:
            cleaned = line_s.replace('];', '').strip()
            if cleaned:
                table3_lines.append([int(x) for x in cleaned.split()])
            if '];' in line_s:
                current_table = None
        elif current_table == 4:
            cleaned = line_s.replace('];', '').strip()
            if cleaned:
                table4_lines.append([int(x) for x in cleaned.split()])
            if '];' in line_s:
                current_table = None

    table2 = np.array(table2_lines)
    table3 = np.array(table3_lines)
    table4 = np.array(table4_lines)
    return table2, table3, table4

def makeH(K=1024, rate=0.5, matlab_file='matrices/makeH_AR4JA.m'):
    table2, table3, table4 = parse_makeH_tables(matlab_file)
    
    theta = table3[:, 1] # shape (26,)
    phi_cat = np.stack([table3[:, 2:9], table3[:, 9:16], table4[:, 2:9], table4[:, 9:16]], axis=2)
    phi = np.transpose(phi_cat, (0, 2, 1))
    
    row_idx = int(np.log2(K) / 2 - 4) - 1 # 0
    col_idx = 0 # for rate 1/2, num = 1
    M = table2[row_idx, col_idx] # 512
    
    phitbl_idx = int(np.log2(M) - 6) - 1 # 2
    
    i = np.arange(M) # 0 to M-1
    M4 = M // 4 # 128
    
    Pi = {}
    for k in range(26):
        quarter = i // M4 # 0, 1, 2, 3
        term1 = M4 * ((theta[k] + quarter) % 4)
        term2 = (phi[k, quarter, phitbl_idx] + i) % M4
        pi_k = term1 + term2 # 0-indexed column for each row i
        Pi[k + 1] = set((r, int(pi_k[r])) for r in range(M))
        
    I_set = set((r, r) for r in range(M))
    
    # Block rows: 0, 1, 2 (each of height M=512)
    # Block cols: 0, 1, 2, 3, 4 (each of width M=512)
    # H=[Z Z I Z I+Pi1; I I Z I Pi2+Pi3+Pi4; I Pi5+Pi6 Z Pi7+Pi8 I];
    block_specs = [
        # row 0
        [(0, 2, I_set), (0, 4, I_set), (0, 4, Pi[1])],
        # row 1
        [(1, 0, I_set), (1, 1, I_set), (1, 3, I_set), (1, 4, Pi[2]), (1, 4, Pi[3]), (1, 4, Pi[4])],
        # row 2
        [(2, 0, I_set), (2, 1, Pi[5]), (2, 1, Pi[6]), (2, 3, Pi[7]), (2, 3, Pi[8]), (2, 4, I_set)]
    ]
    
    edges = set()
    for block_row in block_specs:
        for br, bc, s in block_row:
            for r, c in s:
                edges.add((br * M + r, bc * M + c))
                
    return edges, 3 * M, 5 * M, M

if __name__ == '__main__':
    edges, num_rows, num_cols, M = makeH(1024, 0.5)
    print(f'Constructed H: shape = ({num_rows}, {num_cols}), edges count = {len(edges)}')
    
    # Load chinn.out
    chinn_lines = open('matrices/AR4JA_r45_4c_128c_r12.chinn.out').readlines()
    chinn_edges = set()
    for line in chinn_lines[:-1]:
        parts = line.strip().split()
        if len(parts) == 3 and float(parts[2]) > 0.5:
            chinn_edges.add((int(parts[0]) - 1, int(parts[1]) - 1))
            
    print(f'chinn.out edges count = {len(chinn_edges)}')
    
    # Check match
    diff1 = edges - chinn_edges
    diff2 = chinn_edges - edges
    print(f'Edges in makeH but not in chinn.out: {len(diff1)}')
    print(f'Edges in chinn.out but not in makeH: {len(diff2)}')
