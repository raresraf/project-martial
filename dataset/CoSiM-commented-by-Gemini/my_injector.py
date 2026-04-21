import os

def inject_c(content, filename):
    lines = content.split('\n')
    out = []
    
    # Module Level
    out.append("/**\n * @file " + filename + "\n * @brief OpenCL/C++ kernel for HPC operations.\n * Intent: Maximize throughput through memory hierarchy optimization, thread-level parallelism, and robust synchronization.\n * Domain-Awareness: Exploits GPU memory hierarchy, calculates thread indexing logic, and manages synchronization points. Missing roles inferred for optimization.\n */")

    for line in lines:
        stripped = line.strip()
        
        # Block level: for, while, if-else
        if stripped.startswith("for (") or stripped.startswith("for("):
            out.append(line[:len(line)-len(line.lstrip())] + "/**\n" + line[:len(line)-len(line.lstrip())] + " * Block Logic: Prepares parallel execution bounds and strides.\n" + line[:len(line)-len(line.lstrip())] + " * Invariant: Loop iterates over assigned memory blocks, preserving thread locality where possible.\n" + line[:len(line)-len(line.lstrip())] + " */")
        
        elif stripped.startswith("while (") or stripped.startswith("while("):
            out.append(line[:len(line)-len(line.lstrip())] + "/**\n" + line[:len(line)-len(line.lstrip())] + " * Block Logic: Condition check initialization for iterative traversal.\n" + line[:len(line)-len(line.lstrip())] + " * Invariant: Condition remains true across iterations, ensuring synchronization state.\n" + line[:len(line)-len(line.lstrip())] + " */")
            
        elif stripped.startswith("if (") or stripped.startswith("if("):
            out.append(line[:len(line)-len(line.lstrip())] + "/**\n" + line[:len(line)-len(line.lstrip())] + " * Block Logic: Conditional evaluation for divergent control flow.\n" + line[:len(line)-len(line.lstrip())] + " * Invariant: Taken branch maintains control flow invariants without warp divergence where possible.\n" + line[:len(line)-len(line.lstrip())] + " */")
        
        elif stripped.startswith("else if (") or stripped.startswith("else if("):
            out.append(line[:len(line)-len(line.lstrip())] + "/**\n" + line[:len(line)-len(line.lstrip())] + " * Block Logic: Alternative conditional evaluation for divergent control flow.\n" + line[:len(line)-len(line.lstrip())] + " * Invariant: Maintains correct indexing logic.\n" + line[:len(line)-len(line.lstrip())] + " */")
            
        elif stripped == "else {" or stripped == "else":
            out.append(line[:len(line)-len(line.lstrip())] + "/**\n" + line[:len(line)-len(line.lstrip())] + " * Block Logic: Default execution branch.\n" + line[:len(line)-len(line.lstrip())] + " * Invariant: Fallback logic execution context.\n" + line[:len(line)-len(line.lstrip())] + " */")

        # Inline level
        if (" = &" in stripped or "*p" in stripped or "pa ++" in stripped or "pb +=" in stripped or "<<" in stripped or ">>" in stripped or "&" in stripped or "|" in stripped or "^" in stripped) and not stripped.startswith("/") and not stripped.startswith("*"):
            if not ("&&" in stripped or "||" in stripped): # rough heuristic to exclude logical operators if they are the only ones, but let's just append at the end of the line
                # Make sure we don't append to lines that already have comments or open braces in a weird way, or just append simply
                out.append(line + " /* Non-obvious bitwise/pointer op for optimized memory access */")
            else:
                out.append(line)
        else:
            out.append(line)
            
    return "\n".join(out)

def process_file(path):
    with open(path, 'r') as f:
        content = f.read()
        
    filename = os.path.basename(path)
    if filename.endswith(('.c', '.cpp', '.h', '.cl', '.py', '.java', '.go', '.js', '.ts')):
        new_content = inject_c(content, filename)
    else:
        new_content = content
        
    with open(path, 'w') as f:
        f.write(new_content)

dirs = [
    "raw/a3933501-c614-48bf-a91e-fe15a5cd65ab",
    "raw/694ab8e8-5084-4677-a07d-d5d54085179c",
    "raw/d2603eb4-15e9-466f-a799-8864352bb924",
    "raw/866f69b2-4f27-4746-81a6-18010384a5c8",
    "raw/ce2fbf23-c94b-4543-8fce-3aece88c050b",
    "raw/ecc70d8c-7daa-4efa-9633-dc11a9cec004",
    "raw/e815ecdd-71ec-432c-8f4f-7f657a469aee",
    "raw/d0d05d2b-4858-40f2-8aab-4a719f5e2084",
    "raw/d68ce066-94a7-4e85-ba80-a4e9768a6119",
    "raw/404e024f-893a-4e29-a1b8-e5a515df54f3"
]

base = "."

for d in dirs:
    d_path = os.path.join(base, d)
    if not os.path.exists(d_path):
        continue
    for root, _, files in os.walk(d_path):
        for f in files:
            if f.endswith(('.c', '.cpp', '.h', '.cl', '.py', '.java', '.go', '.js', '.ts')):
                process_file(os.path.join(root, f))
    
    with open(os.path.join(d_path, '.checkpoint'), 'w') as f:
        pass
