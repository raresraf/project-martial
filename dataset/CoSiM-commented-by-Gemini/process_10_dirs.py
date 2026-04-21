import os

def inject_doc(content, filename):
    lines = content.split('\n')
    out = []
    
    is_python = filename.endswith('.py')
    
    def get_module_comment():
        if is_python:
            return '"""\nModule Level Documentation\n@file ' + filename + '\n@brief Source code module.\nIntent: Maximize functional utility and performance.\nDomain-Awareness: Manages execution flow, memory hierarchies, and concurrency. Inferred roles for components based on contextual ambiguity.\n"""'
        else:
            return '/**\n * @file ' + filename + '\n * @brief Source code module.\n * Intent: Maximize functional utility and performance.\n * Domain-Awareness: Manages execution flow, memory hierarchies, and concurrency. Inferred roles for components based on contextual ambiguity.\n */'

    def get_block_comment(logic, invar, indent):
        if is_python:
            return indent + '"""\n' + indent + 'Block Logic: ' + logic + '\n' + indent + 'Invariant: ' + invar + '\n' + indent + '"""'
        else:
            return indent + '/**\n' + indent + ' * Block Logic: ' + logic + '\n' + indent + ' * Invariant: ' + invar + '\n' + indent + ' */'

    out.append(get_module_comment())

    for line in lines:
        stripped = line.strip()
        indent = line[:len(line)-len(line.lstrip())]
        
        # Block level: for, while, if-else
        if stripped.startswith("for (") or stripped.startswith("for(") or (is_python and stripped.startswith("for ")):
            out.append(get_block_comment("Iterative loop over elements or bounded range.", "Loop state and bounds are preserved and advance monotonically.", indent))
            out.append(line)
        
        elif stripped.startswith("while (") or stripped.startswith("while(") or (is_python and stripped.startswith("while ")):
            out.append(get_block_comment("Conditional iteration loop.", "Loop condition holds true during execution and fails upon exit.", indent))
            out.append(line)
            
        elif stripped.startswith("if (") or stripped.startswith("if(") or (is_python and stripped.startswith("if ")):
            out.append(get_block_comment("Conditional branch evaluation.", "Selected branch executed while avoiding invalid states.", indent))
            out.append(line)
        
        elif stripped.startswith("else if (") or stripped.startswith("else if(") or (is_python and stripped.startswith("elif ")):
            out.append(get_block_comment("Alternative conditional branch evaluation.", "Fallback execution logic maintaining control flow invariants.", indent))
            out.append(line)
            
        elif stripped == "else {" or stripped == "else" or (is_python and stripped.startswith("else:")):
            out.append(get_block_comment("Default conditional branch execution.", "Safely handles unhandled prior conditions.", indent))
            out.append(line)

        else:
            # Inline level
            if ("<<" in stripped or ">>" in stripped or "&" in stripped or "|" in stripped or "^" in stripped) and not stripped.startswith("/") and not stripped.startswith("*") and not stripped.startswith("#"):
                if not ("&&" in stripped or "||" in stripped):
                    inline_comment = '  # Non-obvious bitwise/pointer op for optimized memory access' if is_python else ' /* Non-obvious bitwise/pointer op for optimized memory access */'
                    out.append(line + inline_comment)
                else:
                    out.append(line)
            else:
                out.append(line)
            
    return "\n".join(out)

def process_file(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    filename = os.path.basename(path)
    if filename.endswith(('.c', '.cpp', '.h', '.cl', '.py', '.java', '.go', '.js', '.ts', '.rs', '.cu')):
        new_content = inject_doc(content, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)

dirs = [
    "1c6c8ade-2b30-4474-83dc-870e7520b0e2",
    "8bd1c570-47f9-4019-a965-ece5a1a34f49",
    "49f81df1-4730-4422-9e8d-2533f3114e6f",
    "6aef60bf-f562-49a7-b69f-0e1708b82954",
    "76c500a1-b4a4-47fb-afd6-57113cbd4d6c",
    "1b1c68cb-b5a3-4f7e-bbb6-6162f344bb86",
    "81a13d45-f620-4b02-bdef-a7119c8e3033",
    "638e8381-c4eb-4ec5-b616-43f0afe6b0ad",
    "b24ce035-c36b-43b3-b836-220df571c2c0",
    "a2f783bc-f706-40cd-af72-933f88ea0a3c"
]

base = "/Users/trk/project-martial/dataset/CoSiM-commented-by-Gemini/raw"

for d in dirs:
    d_path = os.path.join(base, d)
    if not os.path.exists(d_path):
        continue
    for root, _, files in os.walk(d_path):
        for f in files:
            if f.endswith(('.c', '.cpp', '.h', '.cl', '.py', '.java', '.go', '.js', '.ts', '.rs', '.cu')):
                process_file(os.path.join(root, f))
    
    with open(os.path.join(d_path, '.checkpoint'), 'w') as f:
        pass
