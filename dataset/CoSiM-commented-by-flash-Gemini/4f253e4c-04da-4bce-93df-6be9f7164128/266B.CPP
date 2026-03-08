/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of a state-reordering process in a queue based on local adjacency rules.
 *
 * Problem Name: Queue at the School (Codeforces 266B)
 * Algorithm: Iterative single-pass swap simulation with reverse traversal. For each time step, the queue is scanned
 *            from right to left. If a boy ('B') is at position `i` and a girl ('G') is at `i+1`, they swap.
 *            The loop counter is adjusted after a swap to ensure correct propagation of movement within a single time step.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the queue length.
 * Space Complexity: O(N) to store the queue elements.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, e.g., <iostream>, <algorithm> (for swap).

using namespace std;

/**
 * @brief Main function to handle input, initiate queue simulation, and print the result.
 * @pre Input consists of an integer 'n' (queue length), 't' (time steps), followed by 'n' characters
 *      ('B' or 'G') representing the initial queue state.
 * @post Prints the final state of the queue after 't' simulation steps.
 */
int main() {
  int n, t, i, j; // 'n': length of the queue, 't': number of time steps, 'i, j': loop counters.
  char a[101];    // C-style character array to store the queue elements (max 100 + 1 for sentinel/null).

  // Functional Utility: Reads the queue length 'n' and the number of time steps 't'.
  scanf("%d %d", &n, &t);

  /**
   * Block Logic: Reads the initial state of the queue into the character array 'a'.
   * Note: The loop uses 1-based indexing for `i`, suggesting `a[1]` to `a[n]` are used.
   *       `scanf("%s", &a[i])` is an unconventional but functional way to read single characters into `a[i]`
   *       if each input is a single character.
   */
  for (i = 1; i <= n; i++) {
    scanf("%s", &a[i]); // Reads the character for the i-th position.
  }

  /**
   * Block Logic: Outer loop for simulating 't' time steps.
   * Invariant: Each iteration of this loop represents one unit of time, during which boys and girls can swap positions.
   */
  for (j = 1; j <= t; j++) {
    /**
     * Block Logic: Inner loop performs a single-pass sweep from right to left to resolve "BG" pairs.
     * Invariant: `i` iterates downwards. If `a[i]` is 'B' and it's not the last element, it swaps with `a[i+1]`.
     *            `i` is then decremented by 2 to re-evaluate the character potentially moved into `a[i]`,
     *            ensuring that only one "BG" swap happens at a given position within a time step.
     */
    for (i = n; i >= 1;) { // Loop iterates from the end of the queue backwards.
      // Pre-condition: Checks if the current element is 'B' and not at the very end of the queue.
      // Invariant: This implies a 'G' is at `a[i+1]` to form a "BG" pair that needs swapping.
      if (a[i] == 'B' && i != n) {
        swap(a[i], a[i + 1]); // Swaps the boy and girl characters.
        i = i - 2;            // Adjusts index to re-evaluate the character that just moved into `i-1` and `i`.
      } else {
        i--; // Moves to the previous character.
      }
    }
  }

  /**
   * Block Logic: Prints the final state of the queue.
   * Invariant: Iterates from `a[1]` to `a[n]` and prints each character.
   */
  for (i = 1; i <= n; i++) {
    cout << a[i]; // Prints the character.
  }
  printf("\n"); // Prints a newline character.

  // Indicates successful program execution.
  return 0;
}
