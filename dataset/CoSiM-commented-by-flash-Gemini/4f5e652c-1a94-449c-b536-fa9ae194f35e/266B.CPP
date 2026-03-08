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
#include <bits/stdc++.h> // Includes most standard C++ libraries, e.g., <iostream>, <string>, <algorithm> (for swap).
#include <string>        // Explicitly include <string> for std::string.

using namespace std;

/**
 * @brief Main function to handle input, initiate queue simulation, and print the result.
 * @pre Input consists of an integer 'n' (queue length), 't' (time steps), followed by a string
 *      of 'B' or 'G' characters representing the initial queue state.
 * @post Prints the final state of the queue after 't' simulation steps.
 */
int main() {
  int i, j, n, t; // 'i, j': loop counters, 'n': length of the queue, 't': number of time steps.
  string s;       // String to store the queue elements.

  // Functional Utility: Reads the queue length 'n', number of time steps 't', and the initial queue string 's'.
  cin >> n >> t >> s;

  /**
   * Block Logic: Outer loop for simulating 't' time steps.
   * Invariant: Each iteration of this loop represents one unit of time, during which boys and girls can swap positions.
   */
  for (i = 1; i <= t; i++) { // Loop for 't' times.

    /**
     * Block Logic: Inner loop performs a single-pass sweep from left to right to resolve "BG" pairs.
     * Invariant: `j` iterates through the queue. If `s[j]` is 'B' and `s[j+1]` is 'G', they swap.
     *            `j` is then incremented by an additional 1 to effectively skip the `j+1` position,
     *            preventing the newly moved boy from being considered for another swap within the same time step.
     */
    for (j = 0; j < n - 1; j++) { // Loop through the string from index 0 to `n-2` to allow `s[j+1]`.

      // Pre-condition: Checks for an adjacent 'B' and 'G' pair.
      if (s[j] == 'B' && s[j + 1] == 'G') {
        swap(s[j], s[j + 1]); // Swaps the boy and girl characters.
        j++;                  // Inline: Skips the next index (the new position of the 'B') to ensure it moves only once.
      }
    }
  }

  // Functional Utility: Prints the final state of the queue.
  cout << s << endl; // Prints the result followed by a newline.

  // Indicates successful program execution.
  return 0;
}

// 666Lucifer666
