/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of a state-reordering process in a queue based on local adjacency rules.
 *
 * Problem Name: Queue at the School (Codeforces 266B)
 * Algorithm: Iterative single-pass swap simulation with forward traversal. For each time step, the queue is scanned
 *            from left to right. If a boy ('B') is followed by a girl ('G'), they swap.
 *            The inner loop counter is adjusted after a swap to ensure that a boy moves only once per time step.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the queue length.
 * Space Complexity: O(N) to store the queue elements.
 */
/// Jos  Luiz da Silva Neto
/// Computer Engineering - Federal University of Itajub
#include <bits/stdc++.h> // Includes most standard C++ libraries, e.g., <iostream>, <string>, <algorithm> (for swap).

using namespace std;

// Macros for common competitive programming constants and utilities.
#define INF 0x3F3F3F3F        // Functional Utility: Defines a large integer value for infinity.
#define LINF 0x3F3F3F3F3F3F3F3FLL // Functional Utility: Defines a large long long value for infinity.
#define pb push_back          // Functional Utility: Shorthand for vector::push_back.
#define mp make_pair          // Functional Utility: Shorthand for std::make_pair.
#define F first               // Functional Utility: Shorthand for accessing the first element of a pair.
#define S second              // Functional Utility: Shorthand for accessing the second element of a pair.

// Typedefs for common competitive programming data structures.
typedef pair<int, int> pii;         // Functional Utility: Shorthand for pair of integers.
typedef vector<int> vi;             // Functional Utility: Shorthand for vector of integers.
typedef vector<pii> vii;            // Functional Utility: Shorthand for vector of pairs of integers.
typedef long long ll;               // Functional Utility: Shorthand for long long.
typedef unsigned long long ull;     // Functional Utility: Shorthand for unsigned long long.

// Debugging macro.
#define D(x) cout << #x << " = " << x << endl; // Functional Utility: Macro for printing variable name and value.

// Arrays for common directional movements (e.g., in grid problems). Not used in this specific problem.
int vx[] = {1, 0, -1, 0};
int vy[] = {0, 1, 0, -1};

/**
 * @brief Main function to handle input, initiate queue simulation, and print the result.
 * @pre Input consists of an integer 'n' (queue length), 't' (time steps), followed by a string
 *      of 'B' or 'G' characters representing the initial queue state.
 * @post Prints the final state of the queue after 't' simulation steps.
 */
int main() {
  int n, t; // 'n': length of the queue, 't': number of time steps.
  // Functional Utility: Reads the queue length and the number of time steps.
  cin >> n >> t;

  string x; // String to store the queue elements.
  // Functional Utility: Reads the initial state of the queue as a string.
  cin >> x;

  /**
   * Block Logic: Outer loop for simulating 't' time steps.
   * Invariant: Each iteration of this loop represents one unit of time, during which boys and girls can swap positions.
   */
  while (t--) {
    /**
     * Block Logic: Inner loop performs a single-pass sweep from left to right to resolve "BG" pairs.
     * Invariant: `i` iterates through the queue. If `x[i]` is 'B' and `x[i+1]` is 'G', they swap.
     *            `i` is then incremented by an additional 1 to effectively skip the `i+1` position,
     *            preventing the newly moved boy from being considered for another swap within the same time step.
     */
    for (int i = 0; i < n - 1; i++) { // Loop through the string from index 0 to `n-2` to allow `x[i+1]`.
      // Pre-condition: Checks for an adjacent 'B' and 'G' pair.
      if (x[i] == 'B' && x[i + 1] == 'G') {
        swap(x[i], x[i + 1]); // Swaps the boy and girl characters.
        i++;                  // Inline: Skips the next index (the new position of the 'B') to ensure it moves only once.
      }
    }
  }

  // Functional Utility: Prints the final state of the queue.
  cout << x << endl; // Prints the result followed by a newline.
  // Indicates successful program execution.
  return 0;
}