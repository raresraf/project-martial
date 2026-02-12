/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem.
 * @details This program simulates the reordering of a queue of boys ('B') and girls ('G') over a period of `t` seconds.
 * In each second, any boy standing immediately in front of a girl swaps places with her. This process is repeated
 * for `t` seconds to find the final arrangement of the queue.
 *
 * @algorithm Discrete-time simulation of a swapping process.
 * @complexity
 * Time Complexity: O(T * N), where T is the number of seconds and N is the length of the queue. The outer loop runs T times and the inner loop runs N times.
 * Space Complexity: O(N) to store the queue as a string.
 */
#include <iostream>
#include <string>

int main() {
  // n: number of children in the queue.
  // t: time in seconds for the simulation.
  int n, t;
  std::string queue;

  std::cin >> n >> t;
  std::cin >> queue;

  /**
   * @brief Outer loop to simulate the passage of time.
   * @invariant This loop runs `t` times, with each iteration representing one second.
   */
  for (int j = 0; j < t; j++) {
    /**
     * @brief Inner loop to perform swaps within a single second.
     * @invariant Iterates through the queue to find adjacent 'B' and 'G' pairs.
     */
    for (int i = 0; i < n - 1; i++) {
      // Check if a boy is standing directly in front of a girl.
      if ('B' == queue[i] && 'G' == queue[i + 1]) {
        // If so, swap their positions.
        queue[i] = 'G';
        queue[i + 1] = 'B';
        // Increment `i` an extra time to skip the new position of the boy.
        // This ensures a child moves at most once per second.
        i++;
      }
    }
  }

  // Print the final state of the queue after t seconds.
  std::cout << queue << std::endl;

  return 0;
}