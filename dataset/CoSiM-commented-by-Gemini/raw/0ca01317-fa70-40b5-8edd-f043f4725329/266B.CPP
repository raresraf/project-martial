/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem.
 * @details This program simulates the change in a queue of boys ('B') and girls ('G') over `t` seconds.
 * In each second, any boy standing immediately to the left of a girl will swap places with her.
 * The simulation runs for `t` seconds to produce the final arrangement.
 *
 * @author Conor
 * @date 5/28/2020
 *
 * @algorithm Discrete-time simulation of a swapping process.
 * @complexity
 * Time Complexity: O(T * N), where T is the time in seconds and N is the number of children in the queue.
 * Space Complexity: O(N) to store the string representing the queue.
 */

#include <iostream>

using namespace std;

int main() {
  // n: number of children, t: time in seconds.
  int n, t;
  cin >> n >> t;

  string order;
  cin >> order;

  /**
   * @brief Outer loop representing the passage of time.
   * @invariant Runs for `t` iterations, each representing one second of the simulation.
   */
  for (int i = 0; i < t; ++i) {
    /**
     * @brief Inner loop to process swaps within a single second.
     * @invariant Iterates through the queue to find 'B' followed by 'G'.
     * @note The loop condition `j < n` could lead to an out-of-bounds access at `order[j + 1]`
     * when `j` is `n-1`. However, it often works due to how memory is laid out and the null
     * terminator, but `j < n - 1` would be safer.
     */
    for (int j = 0; j < n; ++j) {
      if (order[j] == 'B' && order[j + 1] == 'G') {
        // Swap the boy and the girl.
        order[j] = 'G';
        order[j + 1] = 'B';
        // Increment `j` to prevent the same boy from moving again in this time step.
        j++;
      }
    }
  }

  // Print the final arrangement of the queue.
  cout << order;

  return 0;
}
