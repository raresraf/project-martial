/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem.
 * @details This program simulates the reordering of a queue of boys ('B') and girls ('G') over `t` seconds.
 * In each second, any boy standing immediately in front of a girl swaps places with her. This process
 * is repeated for `t` seconds to find the final arrangement of the queue.
 *
 * @algorithm Discrete-time simulation of a swapping process.
 * @complexity
 * Time Complexity: O(T * N), where T is the time in seconds and N (`len`) is the length of the queue.
 * Space Complexity: O(N) to store the queue string.
 */
#include <algorithm> // for std::swap
#include <iostream>
#include <stdlib.h> //exit - not used

using namespace std;

int main() {
  // len: the length of the queue.
  // t: the time in seconds for the simulation.
  int len, t;
  string que;
  cin >> len >> t;
  cin >> que;

  /**
   * @brief Outer loop to simulate the passage of time.
   * @invariant Runs for `t` iterations, each representing one second.
   */
  for (int i = 0; i < t; i++) {
    /**
     * @brief Inner loop to perform swaps within a single second.
     * @invariant Iterates through the queue to find adjacent 'B' and 'G' pairs.
     * @note The loop condition `j < len` could lead to an out-of-bounds access at `que[j + 1]`
     * when `j` is `len-1`. `j < len - 1` would be a safer condition.
     */
    for (int j = 0; j < len; j++) {
      if (que[j] == 'B' && que[j + 1] == 'G') {
        // Use std::swap for a clean and efficient swap of characters.
        swap(que[j], que[j + 1]);
        // Increment `j` to prevent the same boy from moving again in this time step.
        j++;
      }
    }
  }
  cout << que;
}