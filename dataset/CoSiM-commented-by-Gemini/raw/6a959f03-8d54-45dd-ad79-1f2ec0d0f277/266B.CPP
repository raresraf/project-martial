
#include <bits/stdc++.h>
using namespace std;

/**
 * @file 266B.CPP
 * @brief Solution to Codeforces problem 266B, "Queue at the School".
 * @details This program simulates the state changes in a queue of boys ('B')
 * and girls ('G') over a given period of time.
 *
 * Algorithm:
 * The simulation proceeds in discrete time steps. In each step, the program
 * iterates through the queue and swaps any adjacent pair of a boy followed by
 * a girl ('BG' -> 'GB'). This process is repeated for the total number of
 * specified time steps.
 *
 * Time Complexity: O(T * N), where T is the number of time steps and N is the
 * number of children in the queue.
 * Space Complexity: O(N) to store the queue configuration.
 */

// Forward declaration of a manual swap function.
void swap(char *x, char *y);

// Global variable, unused in main logic.
int i = 0;

/**
 * @brief Swaps two characters using pointers.
 * @param x Pointer to the first character.
 * @param y Pointer to the second character.
 */
void swap(char *x, char *y) {
  char t;
  t = *x;
  *x = *y;
  *y = t;
}

int main() {
  int i, t, n;
  char a[9999]; // Character array to hold the queue.

  // n: number of children, t: number of time steps.
  cin >> n >> t;
  for (int j = 0; j < n; j++) {
    cin >> a[j];
  }

  /**
   * @brief Main simulation loop that advances time.
   *
   * Invariant: At the start of each iteration, `a` holds the state of the
   * queue at the beginning of the current time step.
   */
  for (int l = t; l >= 1; l--) {
    /**
     * @brief Single-pass sweep to resolve 'BG' pairs for the current time step.
     *
     * Pre-condition: `a` is the queue state at the start of this step.
     * Logic: Identifies adjacent 'B' and 'G' pairs and swaps their positions.
     */
    for (i = 0; i < n; i++) {
      // Note: The check `i+1` can go out of bounds if the last char is 'B'.
      if ((a[i] == 'B') && (a[i + 1] == 'G')) {
        swap(a[i], a[i + 1]); // This call is syntactically incorrect, should be &a[i], &a[i+1].
                               // It likely relies on a compiler extension or std::swap being found.
        /**
         * Inline: Skips the next index. This ensures that after a swap
         * ('BG' -> 'GB'), the newly moved 'B' is not considered for another
         * swap in the same time step, correctly modeling simultaneous swaps.
         */
        i++;
      }
    }
  }

  // Output the final state of the queue after `t` seconds.
  for (int k = 0; k < n; k++) {
    cout << a[k];
  }
  return 0;
}
