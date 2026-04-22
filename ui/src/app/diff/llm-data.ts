export const LLM_DEFAULT_PROMPT = `Role: You are a Senior Software Documentation Engineer and you can perform algorithm analysis at expert level. Your goal is to analyze source code from this dataset, spanning from open-source projects that run for production systems, student assignments, and competitive programming, and augment it with dense semantic documentation. There are many languages that you will find: Go, TypeScript, C, Java, Rust, CUDA, Python, OpenCL, C++.

Objective: Generate human-readable comments (in English) for the source code that could later serve as a "semantic fingerprint" for code similarity analysis. Add the comments to the source files using a syntax specific to the programming language (e.g., // for C/C++, # for Python).

Rules and Constraints:
1. Core change management constraints:
   - Zero Code Mutation: Do not alter, reformat, or "fix" the executable code.
   - Intent-First Documentation: Avoid descriptive syntax (e.g., "declares an integer"). Focus on functional utility (e.g., "acts as a synchronization barrier for the reduction kernel").

2. Domain-Specific Awareness:
   - HPC & Parallelism (CUDA/OpenCL): Document memory hierarchy usage (Shared vs. Global), thread indexing logic, and synchronization points. Explain the tiling strategy if used.
   - Performance Optimization (C/Rust): Identify and comment on optimization techniques such as loop unrolling, cache-friendly data access patterns, or SIMD vectorization.
   - Competitive Programming (C++/Python): Document the underlying algorithm (e.g., "Dynamic Programming with state compression") and the time complexity (e.g., O(N log N)) where applicable.
   - Production Systems (Go/TypeScript): Focus on architectural intent, error handling patterns, and interface implementations.

3. Thriving in Ambiguity:
   - If variable names are non-descriptive (e.g., v1, v2), infer their role from the data flow and document their actual purpose in the logic.
   - If function names are non-descriptive (e.g., f1, f2), infer their inputs, outputs, and core logic and document it.

4. Comment Granularity:
   - Module Level: A header comment explaining the high-level purpose of the file and each important function.
   - Block Level: Before every major for, while, and if-else block, explain the pre-condition and the invariant.
   - Inline Level: Use only for "non-obvious" bitwise operations or pointer arithmetic.

5. Language-Agnostic Semantic Style:
   - Use standardized technical English. This ensures that similarity models (like Universal Sentence Encoder) can effectively compare the semantic content across different programming languages.

6. General Standards:
   - Prioritize language-specific documentation standards (JSDoc for TS, Doxygen for C++, PEP 257 for Python) to transform raw comments into structured metadata.

Workflow: Make all changes inline. Touch all files that have code snippets. Once done with a directory, leave a .checkpoint file to mark completion. Clear context after each directory.`;

export const LLM_DEMO_LEFT = `#include <bits/stdc++.h>
using namespace std;
#define ll long long int
#define MAX (ll)(10e6 + 5)

int main() {
    int t;
    cin >> t;
    while (t--) {
        int n;
        string s;
        cin >> n >> s;
        for (int i = 0; i < n; i++) {
            if (s[i] == 'B' && s[i + 1] == 'G') {
                s[i] = 'G'; s[i + 1] = 'B'; i++;
            }
        }
        cout << s << "\\n";
    }
    return 0;
}`;

export const LLM_DEMO_RIGHT = `import sys
input = sys.stdin.readline

def solve():
    n = int(input())
    s = list(input().strip())
    i = 0
    while i < n - 1:
        if s[i] == 'B' and s[i+1] == 'G':
            s[i], s[i+1] = s[i+1], s[i]
            i += 1
        i += 1
    print(''.join(s))

t = int(input())
for _ in range(t):
    solve()`;

export function computeAnnotationColors(code: string): string[] {
  const lines = code.split(/\r?\n/);
  let inDocstring = false;

  return lines.map(line => {
    const t = line.trim();
    const dqCount = (t.match(/"""/g) ?? []).length;
    const sqCount = (t.match(/'''/g) ?? []).length;
    const toggled = (dqCount + sqCount) % 2 !== 0;

    const wasIn = inDocstring;
    if (toggled) inDocstring = !inDocstring;

    // Inside a docstring block, or line that opens/closes one
    if (wasIn || toggled) return '#E8F5E9';
    // Same-line open+close docstring (even count, e.g. """text""")
    if (dqCount > 0 || sqCount > 0) return '#E8F5E9';
    // C/C++ block and line comments
    if (t.startsWith('//') || t.startsWith('/*') || t.startsWith('*')) return '#E8F5E9';
    // Standalone # comment lines (Python, shell, etc.)
    if (t.startsWith('#')) return '#E8F5E9';
    // Inline comment heuristic: code followed by two spaces then #
    if (line.includes('  #')) return '#E8F5E9';

    return '#FDFDFD';
  });
}

export const LLM_ANNOTATED_LEFT: { [model: string]: string } = {
  'gemini-1.5-pro': `/**
 * @file QueueSimulation.cpp
 * @brief Discrete-time simulation of a state-reordering process based on local adjacency rules.
 *
 * Algorithm: Iterative single-pass swap simulation (Bubble Sort variant restricted to one pass).
 * Time Complexity: O(T * N) where T = number of test cases, N = queue string length.
 * Space Complexity: O(N) for the mutable queue state string.
 */
#include <bits/stdc++.h>
using namespace std;
// Alias: long long integer type shorthand for competitive programming.
#define ll long long int
// Upper bound for large array allocations; set to approximately 10 million.
#define MAX (ll)(10e6 + 5)

int main() {
    int t;
    cin >> t;
    /**
     * Block Logic: Orchestrates temporal progression of the simulation.
     * Invariant: At the start of each iteration, s represents the queue
     * state at time step T-k, where k is the number of completed iterations.
     */
    while (t--) {
        int n;
        string s;
        cin >> n >> s;
        /**
         * Block Logic: Single-pass sweep to resolve all adjacent priority inversions.
         * Pre-condition: s[0..n-1] contains 'B' (Boy) and 'G' (Girl) characters.
         * Identifies adjacent (B, G) pairs and performs a local swap to advance each G.
         */
        for (int i = 0; i < n; i++) {
            if (s[i] == 'B' && s[i + 1] == 'G') {
                s[i] = 'G'; s[i + 1] = 'B';
                i++; // Inline: skips swapped position to enforce atomic one-step movement per time unit.
            }
        }
        cout << s << "\\n";
    }
    return 0;
}`,

  'gemini-2.0-flash': `/**
 * @brief Queue reordering via single-pass adjacent swaps. O(T * N) time.
 */
#include <bits/stdc++.h>
using namespace std;
// Convenience type alias for long long integer.
#define ll long long int
// Maximum array size for competitive programming constraints (~10^7).
#define MAX (ll)(10e6 + 5)

int main() {
    int t;
    cin >> t;
    // Outer loop: processes each independent test case sequentially.
    while (t--) {
        int n;
        string s;
        cin >> n >> s;
        // Inner sweep: resolves all BG inversions in one left-to-right pass.
        for (int i = 0; i < n; i++) {
            // Swap adjacent B-G pair to advance G one position toward the front.
            if (s[i] == 'B' && s[i + 1] == 'G') {
                s[i] = 'G'; s[i + 1] = 'B';
                i++; // Skip swapped index to maintain atomic movement semantics.
            }
        }
        cout << s << "\\n";
    }
    return 0;
}`,

  'gpt-4o': `/**
 * @file solution.cpp
 * @brief Simulates one discrete time step of a queue reordering process.
 *
 * Models a queue of Boys ('B') and Girls ('G') where each Boy immediately
 * preceding a Girl yields their position in a single time step.
 * Uses a greedy single-pass approach: left-to-right sweep swapping each
 * adjacent (B, G) pair exactly once.
 *
 * Time Complexity:  O(T * N) — linear per test case
 * Space Complexity: O(N)     — in-place string manipulation
 */
#include <bits/stdc++.h>
using namespace std;

// Type alias to reduce verbosity of long long declarations in competitive code.
#define ll long long int
// Practical upper bound for problem input constraints (~10 million elements).
#define MAX (ll)(10e6 + 5)

int main() {
    int t;
    cin >> t;

    // Process each test case independently; state resets per iteration.
    while (t--) {
        int n;
        string s;
        cin >> n >> s;

        /**
         * Single left-to-right sweep over the queue string.
         * Invariant: all B-G inversions at indices [0, i-1] have been resolved.
         * The post-swap i++ enforces atomicity: each B advances at most one
         * position per time step, preventing cascade effects.
         */
        for (int i = 0; i < n; i++) {
            if (s[i] == 'B' && s[i + 1] == 'G') {
                s[i] = 'G';
                s[i + 1] = 'B';
                i++; // Prevents double-processing of the swapped B element.
            }
        }
        cout << s << "\\n";
    }
    return 0;
}`,

  'claude-3-5-sonnet': `/**
 * @brief Discrete-time queue simulation: resolves B-before-G inversions in one pass.
 *
 * Models a queue where adjacent Boy ('B') / Girl ('G') pairs swap each time step.
 * Single-pass greedy is correct: each element moves at most one position per step.
 * Complexity: O(T * N).
 */
#include <bits/stdc++.h>
using namespace std;
#define ll long long int
#define MAX (ll)(10e6 + 5)

int main() {
    int t;
    cin >> t;
    while (t--) {
        int n;
        string s;
        cin >> n >> s;
        /**
         * Sweep left-to-right, swapping every adjacent (B, G) pair once.
         * The i++ after a swap ensures each B advances exactly one step,
         * preventing a single B from "bubbling" past multiple Gs in one pass.
         */
        for (int i = 0; i < n; i++) {
            if (s[i] == 'B' && s[i + 1] == 'G') {
                s[i] = 'G'; s[i + 1] = 'B';
                i++; // Atomic movement: skip to prevent cascade swaps this step.
            }
        }
        cout << s << "\\n";
    }
    return 0;
}`,

  'claude-3-opus': `/**
 * @file QueueSimulation.cpp
 * @brief Simulates one discrete time step of a priority-based queue reordering.
 *
 * Models a physical queue of Boys (B) and Girls (G) under the rule that in each
 * time step, every Boy immediately preceding a Girl yields their position.
 * The greedy single-pass approach is correct because:
 *   1. Each B-G swap is local and does not affect previously processed indices.
 *   2. The i++ post-swap enforces the atomicity constraint: a single Boy may
 *      advance at most one position per time step.
 *
 * Algorithm: Greedy Single-Pass Swap (one pass of Bubble Sort)
 * Time Complexity:  O(T * N) — T test cases, each requiring O(N) sweep
 * Space Complexity: O(N)     — in-place mutation of the queue string
 */
#include <bits/stdc++.h>
using namespace std;

// Concise alias for long long int; standard in competitive programming.
#define ll long long int
// Defines the practical upper bound for input sizes in competitive constraints.
#define MAX (ll)(10e6 + 5)

int main() {
    int t;
    cin >> t;

    /**
     * Outer Loop — Test Case Processor
     * Pre-condition:  t > 0; each case provides (n, s) via stdin.
     * Invariant:      After each iteration, s reflects the queue state after
     *                 exactly one simulated discrete time step.
     */
    while (t--) {
        int n;
        string s;
        cin >> n >> s;

        /**
         * Inner Sweep — Priority Inversion Resolver
         * Pre-condition:  s[0..n-1] is a string of 'B' and 'G' characters.
         * Invariant:      At index i, all B-G inversions in s[0..i-1] are resolved.
         * Post-condition: All adjacent B-G pairs have been swapped exactly once.
         */
        for (int i = 0; i < n; i++) {
            if (s[i] == 'B' && s[i + 1] == 'G') {
                s[i] = 'G';
                s[i + 1] = 'B';
                // Advance past swapped B to maintain atomicity:
                // prevents a single B from migrating more than one position per step.
                i++;
            }
        }
        cout << s << "\\n";
    }
    return 0;
}`,
};

export const LLM_ANNOTATED_RIGHT: { [model: string]: string } = {
  'gemini-1.5-pro': `"""
Module: queue_simulation.py
Purpose: Discrete-time simulation of queue reordering via adjacent priority swaps.
         Functionally equivalent to QueueSimulation.cpp — same algorithm, Python syntax.

Algorithm: Iterative single-pass swap simulation.
Time Complexity: O(T * N) — T test cases, N queue length per case.
Space Complexity: O(N) for the mutable character list.
"""
import sys
input = sys.stdin.readline

def solve():
    """
    Reads one test case and advances the queue one discrete time step.

    Performs a single left-to-right sweep, swapping each adjacent (B, G) pair
    to resolve priority inversions. The i += 1 after a swap enforces atomicity:
    each Boy moves at most one position per time step.

    Prints the resulting queue configuration to stdout.
    """
    n = int(input())
    # Convert immutable string to mutable list for O(1) index-based swaps.
    s = list(input().strip())
    i = 0
    # Sweep left-to-right: resolve all BG inversions in one O(N) pass.
    # Invariant: all positions [0, i-1] are free of unresolved B-G inversions.
    while i < n - 1:
        # Swap adjacent B-G pair: Girl advances one step toward queue front.
        if s[i] == 'B' and s[i+1] == 'G':
            s[i], s[i+1] = s[i+1], s[i]
            i += 1  # Inline: atomic movement — prevents cascade displacement.
        i += 1
    print(''.join(s))

# Read total test case count and dispatch each to the solver.
t = int(input())
for _ in range(t):
    solve()`,

  'gemini-2.0-flash': `# queue_simulation.py — Greedy queue reordering, single-pass. O(T*N).
import sys
input = sys.stdin.readline

def solve():
    """Advance the queue one time step: swap each adjacent (B, G) pair once."""
    n = int(input())
    s = list(input().strip())  # Mutable list for in-place character swaps.
    i = 0
    # Left-to-right sweep to resolve all BG inversions in one O(N) pass.
    while i < n - 1:
        if s[i] == 'B' and s[i+1] == 'G':
            s[i], s[i+1] = s[i+1], s[i]
            i += 1  # Atomic step: skip swapped B to prevent cascade.
        i += 1
    print(''.join(s))

# Process each test case independently.
t = int(input())
for _ in range(t):
    solve()`,

  'gpt-4o': `"""
queue_simulation.py

Simulates discrete-time queue reordering where Boys ('B') yield positions to
adjacent Girls ('G') in each time step. Functionally equivalent to the C++ version,
differing only in syntax and I/O idioms.

Time Complexity:  O(T * N)
Space Complexity: O(N) — list conversion of the immutable input string
"""
import sys

# Rebind to faster buffered stdin for competitive programming I/O throughput.
input = sys.stdin.readline


def solve():
    """
    Processes one test case: reads queue state, simulates one time step.

    Performs a single left-to-right sweep. When a Boy ('B') immediately precedes
    a Girl ('G'), they swap. The post-swap i += 1 enforces atomicity: each Boy
    advances at most one position per step.

    Outputs the final queue state after the single time step simulation.
    """
    n = int(input())
    s = list(input().strip())  # Convert to list for O(1) index-based mutation.
    i = 0

    # Sweep over queue resolving B-G inversions one by one.
    # Invariant: all positions [0, i-1] are free of unresolved B-G pairs.
    while i < n - 1:
        if s[i] == 'B' and s[i + 1] == 'G':
            # Advance G one step toward the front; B yields its position.
            s[i], s[i + 1] = s[i + 1], s[i]
            i += 1  # Skip the just-placed B to prevent double-processing.
        i += 1
    print(''.join(s))


t = int(input())
# Each test case is independent; solve sequentially.
for _ in range(t):
    solve()`,

  'claude-3-5-sonnet': `"""
Discrete-time queue simulation: resolves B-before-G inversions.

Same greedy single-pass algorithm as the C++ implementation.
Each Boy ('B') moves at most one position per call to solve().
"""
import sys
input = sys.stdin.readline

def solve():
    """Read queue state, advance one discrete time step, print result."""
    n = int(input())
    s = list(input().strip())
    i = 0
    # Single left-to-right pass: swap each adjacent (B, G) pair.
    # Invariant: s[0..i-1] contains no unresolved B-G inversions.
    while i < n - 1:
        if s[i] == 'B' and s[i+1] == 'G':
            s[i], s[i+1] = s[i+1], s[i]
            i += 1  # Atomic step: skip the swapped B position.
        i += 1
    print(''.join(s))

t = int(input())
for _ in range(t):
    solve()`,

  'claude-3-opus': `"""
Module: queue_simulation.py
Purpose: Python implementation of the discrete-time queue reordering simulation.

Implements the same algorithm as QueueSimulation.cpp, demonstrating the
language-agnostic nature of the underlying computational intent. Identical logic
expressed in two languages enables cross-language similarity detection via the
Universal Sentence Encoder operating on the generated semantic documentation.

Algorithm: Greedy Single-Pass Swap (functionally equivalent to one Bubble Sort pass)
Time Complexity:  O(T x N) — T test cases, each requiring one O(N) sweep
Space Complexity: O(N)     — list conversion of the immutable input string
"""
import sys

# Rebind input to buffered readline for improved I/O performance under competition constraints.
input = sys.stdin.readline


def solve():
    """
    Processes one test case of the queue simulation protocol.

    Reads the initial queue configuration and simulates exactly one time step
    via a single left-to-right sweep. Each adjacent (Boy, Girl) pair encountered
    is swapped to enforce the priority-yielding rule.

    The i += 1 post-swap maintains the atomicity invariant: each Boy may advance
    at most one position per time step, preventing cascade displacement that would
    violate the discrete-time model.

    Writes the resulting queue configuration to stdout after sweep completion.
    """
    n = int(input())
    # Convert the immutable string to a mutable list for O(1) element swaps.
    s = list(input().strip())
    i = 0

    # Priority Inversion Resolver — left-to-right sweep of the queue.
    # Pre-condition:  s[0..n-2] contains 'B' and 'G' characters.
    # Invariant:      At position i, all B-G inversions in s[0..i-1] are resolved.
    # Post-condition: Every adjacent (B, G) pair has been swapped exactly once.
    while i < n - 1:
        if s[i] == 'B' and s[i + 1] == 'G':
            # Swap: Girl advances one position toward the front of the queue.
            s[i], s[i + 1] = s[i + 1], s[i]
            # Advance past the swapped Boy to preserve atomicity of movement.
            i += 1
        i += 1
    print(''.join(s))


# Dispatch each independent test case to the solver function.
t = int(input())
for _ in range(t):
    solve()`,
};
