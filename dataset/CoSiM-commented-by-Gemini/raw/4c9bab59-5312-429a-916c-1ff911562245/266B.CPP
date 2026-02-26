/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem from Codeforces.
 * @details This program simulates the arrangement of a queue of boys ('B') and
 * girls ('G') over a given period of time. In each second, any boy standing
 * directly in front of a girl swaps places with her. This process is repeated
 * for a specified number of seconds.
 *
 * @algorithm
 * The program performs a direct, discrete-time simulation.
 * 1. It iterates for a total of `t` time steps (seconds).
 * 2. In each time step, it scans the queue from left to right.
 * 3. If it finds a 'B' followed immediately by a 'G', it swaps them.
 * 4. To ensure that swaps within a single time step are effectively simultaneous
 *    and a newly moved character isn't processed again in the same step, the
 *    loop counter is advanced an extra position after a swap.
 *
 * @complexity
 * Time Complexity: O(T * N), where T is the number of seconds and N is the
 * number of children in the queue.
 * Space Complexity: O(N) to store the queue arrangement.
 */
#include <iostream>
#include <vector>   // Using std::vector is safer than raw pointers.
#include <string>   // For std::string usage.

/**
 * @brief Simulates the position changes in the queue over `t` seconds.
 * @param values A reference to the vector representing the queue.
 * @param n The size of the queue.
 * @param t The number of seconds to simulate.
 */
void changePositions(std::vector<char>& values, int n, int t) {
  // The outer loop simulates the passage of time.
  while (t > 0) {
    // The inner loop scans the queue for possible swaps in one time step.
    for (int i = 0; i < n - 1; ++i) {
      if ((values[i] == 'B') && (values[i + 1] == 'G')) {
        // A boy is in front of a girl, so they swap.
        values[i] = 'G';
        values[i + 1] = 'B';
        /**
         * @inline
         * @brief Advance the index to prevent re-swapping.
         * @details This is a crucial step. It ensures that after a 'B' moves
         * right and a 'G' moves left, we continue the scan from the position
         * *after* the swapped pair. This correctly models the simultaneous nature
         * of swaps within a single second.
         */
        ++i;
      }
    }
    --t;
  }
}

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(NULL);

  int n, t;
  std::cin >> n >> t;

  // Using std::vector is safer and manages memory automatically.
  std::vector<char> values(n);
  for (int i = 0; i < n; ++i) {
    std::cin >> values[i];
  }

  // Perform the simulation.
  changePositions(values, n, t);

  // Print the final queue arrangement.
  for (int i = 0; i < n; ++i) {
    std::cout << values[i];
  }
  std::cout << '\n';

  // No need to `delete[] values` when using std::vector.
  return 0;
}