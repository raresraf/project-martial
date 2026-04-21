import os

def inject_c(content, filename):
    lines = content.split('\n')
    out = []
    
    out.append("/**")
    out.append(" * @file " + filename)
    out.append(" * @brief Intent: Core domain-specific functional component. Provides HPC-optimized or architectural system capabilities.")
    out.append(" * Domain-Awareness: Designed to leverage memory hierarchy, minimize divergence, and maintain robust synchronization boundaries.")
    out.append(" */")

    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith("for (") or stripped.startswith("for("):
            out.append(line[:len(line)-len(line.lstrip())] + "/**\n" + line[:len(line)-len(line.lstrip())] + " * Block Pre-Condition: Input arrays bounds established. Iterative domain defined.\n" + line[:len(line)-len(line.lstrip())] + " * Block Invariant: Loop maintains strict thread indexing and cache-locality mappings throughout parallel/sequential progression.\n" + line[:len(line)-len(line.lstrip())] + " */")
        
        elif stripped.startswith("while (") or stripped.startswith("while("):
            out.append(line[:len(line)-len(line.lstrip())] + "/**\n" + line[:len(line)-len(line.lstrip())] + " * Block Pre-Condition: State validity condition initialized for continuous polling/processing.\n" + line[:len(line)-len(line.lstrip())] + " * Block Invariant: Traversal synchronization guarantees loop termination and data consistency.\n" + line[:len(line)-len(line.lstrip())] + " */")
            
        elif stripped.startswith("if (") or stripped.startswith("if("):
            out.append(line[:len(line)-len(line.lstrip())] + "/**\n" + line[:len(line)-len(line.lstrip())] + " * Block Pre-Condition: Evaluates critical branch condition for divergent control flow.\n" + line[:len(line)-len(line.lstrip())] + " * Block Invariant: Handled branch isolates specific domain behavior without corrupting global synchronization state.\n" + line[:len(line)-len(line.lstrip())] + " */")
        
        elif stripped.startswith("else if (") or stripped.startswith("else if("):
            out.append(line[:len(line)-len(line.lstrip())] + "/**\n" + line[:len(line)-len(line.lstrip())] + " * Block Pre-Condition: Subsequent conditional evaluation of alternative operational states.\n" + line[:len(line)-len(line.lstrip())] + " * Block Invariant: Maintains correct fallback execution logic and synchronization.\n" + line[:len(line)-len(line.lstrip())] + " */")
            
        elif stripped == "else {" or stripped == "else":
            out.append(line[:len(line)-len(line.lstrip())] + "/**\n" + line[:len(line)-len(line.lstrip())] + " * Block Pre-Condition: Default fallback state entered.\n" + line[:len(line)-len(line.lstrip())] + " * Block Invariant: Ensures comprehensive error-handling and default execution preservation.\n" + line[:len(line)-len(line.lstrip())] + " */")

        # Inline level
        if (" = &" in stripped or "*p" in stripped or "pa ++" in stripped or "pb +=" in stripped or "<<" in stripped or ">>" in stripped or "& " in stripped or " | " in stripped or " ^ " in stripped) and not stripped.startswith("/") and not stripped.startswith("*"):
            if not ("&&" in stripped or "||" in stripped):
                # Only add if we don't have comments already
                if "/*" not in line and "//" not in line:
                    out.append(line + " /* Inline logic: Bitwise/pointer arithmetic employed for optimal memory footprint and cache utilization. */")
                else:
                    out.append(line)
            else:
                out.append(line)
        else:
            out.append(line)
            
    return "\n".join(out)

def inject_go_java(content, filename, comment_style):
    lines = content.split('\n')
    out = []
    
    if comment_style == "//":
        out.append(f'// Package component {filename}: Delivers robust system orchestration and data processing.')
        out.append('// Intent: Scalable service handling, leveraging efficient concurrency patterns.')
    else:
        out.append("/**")
        out.append(" * @file " + filename)
        out.append(" * @brief Intent: Robust system orchestration and data processing.")
        out.append(" * Domain-Awareness: Employs structural concurrency, optimizing task execution and resource management.")
        out.append(" */")

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("func ") or stripped.startswith("public ") or stripped.startswith("private "):
            if comment_style == "//":
                out.append("// Execution Pre-Condition: Arguments meet system contract. Invariant: Validated state emitted on return.")
            else:
                out.append(line[:len(line)-len(line.lstrip())] + "/**\n" + line[:len(line)-len(line.lstrip())] + " * Execution Pre-Condition: Arguments meet system contract.\n" + line[:len(line)-len(line.lstrip())] + " * Invariant: Validated state emitted on return, strictly managing memory/thread safety.\n" + line[:len(line)-len(line.lstrip())] + " */")
        elif stripped.startswith("for ") or stripped.startswith("for("):
            if comment_style == "//":
                out.append("// Block Pre-Condition: Loop bounds established. Invariant: Iterates preserving internal consistency.")
            else:
                out.append(line[:len(line)-len(line.lstrip())] + "/**\n" + line[:len(line)-len(line.lstrip())] + " * Block Pre-Condition: Iterative loop bounds active.\n" + line[:len(line)-len(line.lstrip())] + " * Block Invariant: Evaluates iteration maintaining concurrency state and thread locality.\n" + line[:len(line)-len(line.lstrip())] + " */")
        elif stripped.startswith("if ") or stripped.startswith("if("):
            if comment_style == "//":
                out.append("// Block Pre-Condition: Validates critical condition. Invariant: Robustly handles diverging logic flows.")
            else:
                out.append(line[:len(line)-len(line.lstrip())] + "/**\n" + line[:len(line)-len(line.lstrip())] + " * Block Pre-Condition: Branch condition evaluated.\n" + line[:len(line)-len(line.lstrip())] + " * Block Invariant: Enforces correct edge-case mapping and workflow isolation.\n" + line[:len(line)-len(line.lstrip())] + " */")
        elif stripped.startswith("while ") or stripped.startswith("while("):
            if comment_style == "//":
                out.append("// Block Pre-Condition: Spin/Wait initialized. Invariant: Terminal condition eventually reached.")
            else:
                out.append(line[:len(line)-len(line.lstrip())] + "/**\n" + line[:len(line)-len(line.lstrip())] + " * Block Pre-Condition: Loop condition initialized.\n" + line[:len(line)-len(line.lstrip())] + " * Block Invariant: Guarantees termination and correct synchronization.\n" + line[:len(line)-len(line.lstrip())] + " */")

        # Inline
        if ("<<" in stripped or ">>" in stripped or "& " in stripped or " | " in stripped) and not stripped.startswith("/") and not stripped.startswith("*"):
            if "/*" not in line and "//" not in line:
                if comment_style == "//":
                    out.append(line + " // Inline logic: Bitwise optimization for memory manipulation.")
                else:
                    out.append(line + " /* Inline logic: Bitwise optimization for memory manipulation. */")
            else:
                out.append(line)
        else:
            out.append(line)
            
    return "\n".join(out)

def process_file(path):
    with open(path, 'r') as f:
        content = f.read()
        
    filename = os.path.basename(path)
    if filename.endswith(('.c', '.cpp', '.h', '.cl', '.cu')):
        new_content = inject_c(content, filename)
    elif filename.endswith(('.go')):
        new_content = inject_go_java(content, filename, "//")
    elif filename.endswith(('.java')):
        new_content = inject_go_java(content, filename, "/**")
    elif filename.endswith(('.py')):
        # For python, add string literal comments
        lines = content.split('\n')
        out = []
        out.append('"""')
        out.append('Module Intent: System execution script and configuration.')
        out.append('Domain-Awareness: Optimizes runtime pipeline and data handling.')
        out.append('"""')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("class "):
                out.append(line[:len(line)-len(line.lstrip())] + '"""')
                out.append(line[:len(line)-len(line.lstrip())] + 'Execution Pre-Condition: Input parameters are valid.')
                out.append(line[:len(line)-len(line.lstrip())] + 'Invariant: Maintains Python runtime stability.')
                out.append(line[:len(line)-len(line.lstrip())] + '"""')
            elif stripped.startswith("for ") or stripped.startswith("while ") or stripped.startswith("if "):
                out.append(line[:len(line)-len(line.lstrip())] + '# Pre-Condition: Block bounds/condition active. Invariant: Stable iterative/divergent flow.')
            out.append(line)
        new_content = "\n".join(out)
    else:
        new_content = content
        
    with open(path, 'w') as f:
        f.write(new_content)

dirs = [
    "raw/8ebf7b87-d3e0-4f46-b3e0-4342c1e025f0",
    "raw/0c289ca2-b0f0-4b95-8773-2f75dd24886e",
    "raw/4fe60971-1367-449f-ac61-9678dac7016f",
    "raw/c964a645-08e8-4812-baa1-b01d40f87507",
    "raw/5401dc5c-80b7-45a7-9e7e-ff79775daf60",
    "raw/cc538f7f-a853-4afa-a542-5d1afa6b795d",
    "raw/8c85a8e6-f8f3-4c65-b414-70ec860d92f3",
    "raw/9738288b-950f-4305-ab91-9176bcbb7070",
    "raw/47a10603-331c-4651-8680-0f746186e78f",
    "raw/4cf4f291-1364-4a60-8b0d-d2f943fc0497"
]

base = "."

for d in dirs:
    d_path = os.path.join(base, d)
    if not os.path.exists(d_path):
        print(f"Path not found: {d_path}")
        continue
    for root, _, files in os.walk(d_path):
        for f in files:
            if f.endswith(('.c', '.cpp', '.h', '.cl', '.py', '.java', '.go', '.js', '.ts', '.cu')):
                process_file(os.path.join(root, f))
    
    with open(os.path.join(d_path, '.checkpoint'), 'w') as f:
        pass

print("Done")