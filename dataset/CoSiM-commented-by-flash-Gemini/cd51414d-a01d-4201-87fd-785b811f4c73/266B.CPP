/**
 * @file 266B.CPP
 * @brief Simulates a queue reordering process based on adjacent character swaps.
 *   This program models the movement of 'B' (boys) and 'G' (girls) in a queue,
 *   where 'B' characters attempt to move past 'G' characters.
 * * Algorithm: Iterative single-pass swap simulation.
 *   In each time step (iteration), the queue is scanned, and every adjacent pair
 *   'BG' is swapped to 'GB'. A 'B' can move at most one position forward per time step.
 * Time Complexity: O(T * S) where T is the number of time steps and S is the length of the string.
 * Space Complexity: O(S) for storing the string.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Swaps two character references.
 * @param a The first character to swap.
 * @param b The second character to swap.
 * Functional Utility: Provides a utility for in-place character swapping.
 */
void swap(char &a, char &b) {
  char temp;
  temp = a;
  a = b;
  b = temp;
}

/**
 * @brief Main function to simulate the queue reordering process.
 * Functional Utility: Handles input/output and orchestrates the queue simulation logic.
 */
int main() {
  int s, t, f = 0; // 's' is string length, 't' is time steps. 'f' is unused.
  cin >> s >> t;
  string str;
  cin >> str;

  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * Invariant: At the start of each iteration, the string represents the queue state at time T-k.
   */
  for (int i = 0; i < t; ++i) {
    /**
     * Block Logic: Performs a single-pass sweep to resolve priority inversions ('BG' pairs).
     * Pre-condition: The string 'str' represents the current state of the queue.
     * Invariant: After each full pass, all immediately adjacent 'BG' pairs have been resolved.
     */
    for (int j = 0; j < s; ++j) {
      /**
       * Block Logic: Identifies and performs a local state transition (swap).
       * Condition: If a 'B' is directly followed by a 'G', they swap positions.
       */
      if (str[j] == 'B' && str[j + 1] == 'G') {
        swap(str[j], str[j + 1]);
        // Inline: Skips the next index to prevent a single 'B' from migrating
        // more than one position per time step (atomic movement).
        j++;
      }
    }
  }

  // Output the final state of the queue after all time steps.
  cout << str;
}