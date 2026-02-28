/**
 * @file 266B.CPP
 * @brief Solution for the "Queue at the school" problem from Codeforces.
 * @details This program simulates the changing positions in a queue of boys ('B')
 * and girls ('G'). At each time step, if a boy is standing directly in front of
 * a girl, they swap places. The program calculates the final state of the queue
 * after a given amount of time `t`.
 *
 * @algorithm Discrete Time Simulation
 * The simulation proceeds in discrete time steps.
 * 1. An outer loop runs `t` times, representing each second.
 * 2. An inner loop iterates through the queue for each time step.
 * 3. Inside the inner loop, it looks for adjacent pairs of 'B' followed by 'G'.
 * 4. When a 'BG' pair is found, the characters are swapped.
 * 5. Crucially, after a swap at position `j` and `j+1`, the inner loop's index
 *    is advanced by 2. This is because the newly positioned boy at `j+1` cannot
 *    move again in the same time step, so we can safely skip checking that
 *    position and continue searching for the next independent 'BG' pair.
 * 6. If no swap occurs, the index is advanced by 1.
 * 7. After the outer loop completes, the final queue configuration is printed.
 *
 * @complexity
 * - Time: O(T * N), where T is the time and N is the length of the queue.
 * - Space: O(N), to store the queue string.
 */
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[]) {
  // n: the number of children in the queue.
  // t: the total time in seconds to simulate.
  int n, t;
  cin >> n >> t;
  
  // The string representing the queue of Boys and Girls.
  string Queue;
  cin >> Queue;
  
  /**
   * @brief Outer loop to simulate the passage of time for `t` seconds.
   */
  for (int i = 0; i < t; i++) {
    int j = 0;
    /**
     * @brief Inner loop to scan the queue and perform all possible swaps for the current time step.
     */
    // Invariant: The sub-queue from 0 to `j-1` is in its final state for the current time step `i`.
    while (j < n - 1) {
      // Pre-condition: Check if a boy is standing directly in front of a girl.
      // The `(char)` cast is redundant but harmless.
      if (Queue[j] == (char)'B' && Queue[j + 1] == (char)'G') {
        // If so, swap their positions.
        swap(Queue[j], Queue[j + 1]);
        // After a swap, we can skip ahead two positions. The boy at `j+1` can't
        // move again in this time step, so we can start the next check at `j+2`.
        j += 2;
      } else {
        // If no swap, move to the next position to check the next pair.
        j += 1;
      }
    }
  }
  
  // Print the final arrangement of the queue after `t` seconds.
  cout << Queue;
  
  return 0; // Indicate successful execution.
}
