/*
 ~ Author    : @tridib_2003
 ~ Title     : 266B. Queue at the School
 ~ Link      : https://codeforces.com/contest/266/problem/B
*/

/**
 * @file 266B.CPP
 * @brief Solution to a competitive programming problem: "Queue at the School".
 *
 * This program simulates the reordering of a queue consisting of boys ('B') and girls ('G').
 * In each time step, if a boy is immediately followed by a girl ('BG'), they swap positions
 * to become 'GB'. This process is repeated for a given number of time steps 't'.
 *
 * Algorithm: Discrete-time simulation with a single-pass swap rule.
 * Time Complexity: O(T * N), where T is the number of time steps and N is the length of the queue (string).
 *                  Each time step involves a single pass through the queue.
 * Space Complexity: O(N) to store the queue string.
 */
#include <bits/stdc++.h> // Common header in competitive programming, includes most standard libraries for convenience.

/**
 * @brief Standard namespace usage.
 * In competitive programming, 'using namespace std;' is common practice to avoid
 * repeatedly prefixing standard library elements with 'std::'.
 */
using namespace std;

// Macros for common operations and type definitions.
#define ll long long // Type definition for long long integer.
#define PI 3.1415926535897932384626 // Definition of PI, a mathematical constant.
#define MOD 1000000007 // Constant for modulo operation, frequently used in problems involving large numbers.
#define vi vector<int> // Type definition for a vector of integers.
// Macro to dynamically allocate an array. Note: 'type *arr = new type[n];' is C++ specific for heap allocation.
#define mk(arr, n, type) type *arr = new type[n];
#define FOR(a, c) for (int(a) = 0; (a) < (c); (a)++) // Macro for a common for-loop pattern (0 to c-1).
// Macro for handling multiple test cases. Reads 'x' and then loops 'x' times.
#define w(x)                                                                   \
  int x;                                                                       \
  cin >> x;                                                                    \
  while (x--)
// Macro for fast I/O operations. Disables synchronization with C standard streams and unties cin/cout.
#define FIO                                                                    \
  ios::sync_with_stdio(false);                                                 \
  cin.tie(0);                                                                  \
  cout.tie(0);

/**
 * @brief Main function: entry point of the program.
 * Orchestrates the simulation of the queue reordering process based on the given rules
 * and prints the final state of the queue.
 */
int main() {

  FIO; // Apply fast I/O optimizations.

  int tc = 1; // Declares an integer 'tc' for the number of test cases.
  // cin >> tc; // Commented out, implying this solution is designed for a single test case or specific problem setup.

  /**
   * @brief Loop through each test case.
   * Block Logic: Processes one or more problem instances. In this solution, it runs once.
   * Invariant: For each iteration, the problem's inputs are read and the simulation is executed.
   */
  while (tc--) {

    int n, t; // Declares 'n' for the length of the queue and 't' for the number of time steps.
    cin >> n >> t; // Reads the queue length and the number of time steps.

    string q; // Declares a string 'q' to represent the queue.
    cin >> q; // Reads the initial state of the queue.

    /**
     * @brief Simulates the queue reordering over 't' time steps.
     * Block Logic: Each iteration represents one second in the simulation, applying the swap rule.
     * Invariant: At the start of each iteration, 't' represents the remaining time steps.
     */
    while (t--) {

      /**
       * @brief Performs a single-pass sweep to apply the 'BG' to 'GB' swap rule.
       * Block Logic: Iterates through the queue, identifying and executing 'BG' swaps.
       * Pre-condition: 'q' represents the current state of the queue for the current time step.
       * Invariant: After each swap, the 'B' character has moved one position backward.
       */
      for (int i = 0; i < q.length() - 1; ++i) {
        // Condition for swapping: a 'B' is followed by a 'G'.
        if (q[i] == 'B' && q[i + 1] == 'G') {
          q[i] = 'G';     // The boy moves to the next position.
          q[i + 1] = 'B'; // The girl moves to the current position.
          ++i;            // Inline: Increment 'i' again to skip the newly moved 'B', ensuring it moves only one position per second.
                          // This prevents a single boy from "jumping" multiple girls in one time step.
        }
      }
    }

    cout << q << '\n'; // Prints the final state of the queue after all time steps.
  }

  return 0; // Indicates successful program execution.
}