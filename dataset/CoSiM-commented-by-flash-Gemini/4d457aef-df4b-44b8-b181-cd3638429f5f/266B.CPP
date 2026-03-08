/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of a state-reordering process in a queue based on local adjacency rules.
 *
 * Problem Name: Queue at the School (Codeforces 266B)
 * Algorithm: Iterative single-pass swap simulation. In each time step, adjacent 'B' (boy) and 'G' (girl)
 *            pairs are swapped if 'B' precedes 'G'. This implementation uses `std::string` and `std::swap`.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the queue length.
 * Space Complexity: O(N) to store the queue elements.
 */
// code by Nikhil Nagrale
// nikhilnagrale2 on EveryPlatform
#include <bits/stdc++.h> // Includes most standard C++ libraries, e.g., <iostream>, <string>, <algorithm>.
using namespace std;

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

  string a; // String to store the queue elements.
  // Functional Utility: Reads the initial state of the queue as a string.
  cin >> a;

  /**
   * Block Logic: Outer loop for simulating 't' time steps.
   * Invariant: Each iteration of this loop represents one unit of time, during which boys and girls can swap positions.
   */
  while (t--) {
    /**
     * Block Logic: Performs a single-pass sweep through the queue to resolve priority inversions ('B' followed by 'G').
     * Invariant: In each iteration, 'i' scans the queue. If a "BG" pair is found, they are swapped,
     *            and 'i' is incremented to skip the newly swapped 'B' (which just moved) to prevent
     *            it from moving more than one position in a single time step.
     */
    for (int i = 0; i < n - 1; i++) { // Iterates up to n-2 to access a[i+1].
      // Pre-condition: Checks for an adjacent 'B' and 'G' pair.
      if (a[i] == 'B' && a[i + 1] == 'G') {
        swap(a[i], a[i + 1]); // Swaps the boy and girl characters.
        i++;                  // Inline: Skips the next index to prevent the 'B' from migrating more than one position per time step.
      }
    }
  }

  // Functional Utility: Prints the final state of the queue.
  cout << a << endl;
  // Indicates successful program execution.
  return 0;
}