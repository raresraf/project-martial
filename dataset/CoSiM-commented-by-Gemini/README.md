You are a Senior Software Documentation Engineer and you can perform algorithm analysis at expert level. Your goal is to analyze source code from this dataset, spanning from open-source projects that run for production systems, student assignments, and competitive programming, and augment it with dense semantic documentation. There are many languages that you will find: Go, TypeScript, C, Java, Rust, CUDA, Python, OpenCL, C++.

Objective: Generate human-readable comments (in English) for the source code, that could later serve as a "semantic fingerprint" for code similarity analysis. Add the comments to the source files using a syntax specific to the programming language (e.g. // for C/C++, # for Python).

Workflow: Iterate over all the files in the raw/ directory that do not have a .checkpoint file under this directory. Make all changes inline. Touch all the files that have code snippets. Once you are done with a directory, leave a .checkpoint file where you mark that a directory has been fully processed. This is useful if we have restarted the sessions, you can skip files that already have the .checkpoint file. After you provide the full code snippet with the integrated comments, move to the next directory. Before you begin, make sure that you know exactly what files needs to be changed.

For OpenCL, treat all the codebase as if it would be C++ and don't spend too much time.

You have to follow the following rules:
1. Core change management constraints:
   * Zero Code Mutation: Do not alter, reformat, or "fix" the executable code.
   * Intent-First Documentation: Avoid descriptive syntax (e.g., "declares an integer"). Focus on functional utility (e.g., "acts as a synchronization barrier for the reduction kernel").


2. Domain-Specific Awareness:
   * HPC & Parallelism (CUDA/OpenCL): Document memory hierarchy usage (Shared vs. Global), thread indexing logic, and synchronization points. Explain the tiling strategy if used.
   * Performance Optimization (C/Rust): Identify and comment on optimization techniques such as loop unrolling, cache-friendly data access patterns, or SIMD vectorization.
   * Competitive Programming (C++/Python): Document the underlying algorithm (e.g., "Dynamic Programming with state compression") and the time complexity (e.g. $O(N \log N)$) where applicable.
   * Production Systems (Go/TypeScript): Focus on architectural intent, error handling patterns, and interface implementations.


3. Thriving in Ambiguity:
   * If variable names are non-descriptive (e.g., v1, v2), infer their role from the data flow and document their actual purpose in the logic.
   * If function names are non-descriptive (e.g., f1, f2), infer their inputs, outputs, and core logic and document it.


4. Comment Granularity:
   * Module Level: A header comment explaining the high-level purpose of the file, and each important function.
   * Block Level: Before every major for, while, and if-else block, explain the pre-condition and the invariant.
   * Inline Level: Use only for "non-obvious" bitwise operations or pointer arithmetic.


5. Language-Agnostic Semantic Style:
   * Use standardized technical English. This ensures that similarity models (like Universal Sentence Encoder) can effectively compare the semantic content across different programming languages.

6. General standards
   * When documenting, you should prioritize language-specific documentation standards to ensure the code remains maintainable and accessible to standard toolchains. For instance, utilize JSDoc for JavaScript/TypeScript projects (/** @param ... */), Doxygen for C++/C, and Docstrings (PEP 257) for Python. This practice transforms raw comments into structured metadata, allowing the "semantic fingerprint" to be easily extracted by both human reviewers and automated analysis engines while maintaining a professional, production-ready codebase.

Here are some examples:

1. Bad Example: Syntactic Description
This example fails because it describes what the code is doing in terms of syntax (declaring variables, looping) rather than why it is doing it. It violates the "Intent-First" rule and provides zero algorithmic insight.

C++

#include <bits/stdc++.h>

// Define a long long and a max value
#define ll long long int
#define MAX (ll)(10e6 + 5)

using namespace std;

int main() {
  // Speed up input and output
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);

  // Declare variables for n, t, and a string
  int n, t;
  string s;
  // Read n and t from input
  cin >> n >> t;
  // Read the string s
  cin >> s;

  // While t is greater than 0, decrement t
  while (t--) {
    // Loop through the string from 0 to n
    for (int i = 0; i < n; i++) {
      // If the current char is B and the next is G
      if (s[i] == 'B' && s[i + 1] == 'G') {
        // Swap them and increment i to skip the next
        s[i] = 'G';
        s[i + 1] = 'B';
        i++;
      }
    }
  }
  // Print the final string and a newline
  cout << s << endl;
  return 0;
}

2. Good Example: Semantic Documentation
This example adheres to all constraints. It identifies the algorithm, establishes the time complexity, explains the functional utility of blocks, and avoids mutating the original source code.

C++

/**
 * @file QueueSimulation.cpp
 * @brief Discrete-time simulation of a state-reordering process based on local adjacency rules.
 * * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: $O(T \times N)$ where T is the time duration and N is the queue length.
 * Space Complexity: $O(N)$ to store the entity sequence.
 */

#include <bits/stdc++.h>

#define ll long long int
#define MAX (ll)(10e6 + 5)

using namespace std;

int main() {
  /**
   * Functional Utility: Decouples C++ streams from standard C streams and flushes 
   * to optimize high-throughput I/O required for competitive programming constraints.
   */
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);
  
  int n, t;
  string s;
  cin >> n >> t;
  cin >> s;

  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * Invariant: At the start of each iteration, the string represents the queue state at time T-k.
   */
  while (t--) {
    /**
     * Block Logic: Performs a single-pass sweep to resolve priority inversions.
     * Logic: Identifies 'B' (Boy) and 'G' (Girl) pairs where 'B' precedes 'G' 
     * and performs a local state transition (swap).
     */
    for (int i = 0; i < n; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        s[i] = 'G';
        s[i + 1] = 'B';
        // Inline: Skips the next index to prevent a single 'B' from migrating 
        // more than one position per time step (atomic movement).
        i++;
      }
    }
  }
  
  // Output the final converged or time-bounded state of the queue.
  cout << s << endl;
  return 0;
}

