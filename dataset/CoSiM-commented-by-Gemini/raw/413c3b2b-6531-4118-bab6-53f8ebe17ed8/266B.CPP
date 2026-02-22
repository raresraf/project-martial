/**
 * @file 266B.CPP
 * @brief Solution to Codeforces problem 266B - Queue at the School.
 *
 * @algorithm
 * This program simulates a queue of boys ('B') and girls ('G'). At each time
 * step, if a boy is standing directly in front of a girl, they swap places.
 * This process is repeated for a given number of time steps.
 *
 * The algorithm is as follows:
 * 1. Read the queue length `n`, the total time `b`, and the initial queue string `a`.
 * 2. Loop `b` times to simulate each second passing.
 * 3. Inside the time loop, iterate through the queue string from left to right.
 * 4. If a 'B' is found at index `j` and a 'G' at `j+1`, swap them using `std::swap`.
 * 5. After a swap, increment the loop counter `j` by one extra position. This is
 *    critical to ensure that a person only moves once per time step.
 * 6. After all time steps are simulated, print the final state of the queue.
 *
 * @complexity
 * - Time Complexity: O(b * n), where `b` is the number of time steps and `n` is the length of the queue.
 * - Space Complexity: O(n) to store the queue string `a`.
 */
#include <algorithm>
#include <iostream>
using namespace std;
int n, b;
string a;
int main() {
  cin >> n >> b >> a;
  /**
   * @brief This outer loop simulates the passage of time for `b` seconds.
   */
  for (int i = 0; i < b; i++) {
    /**
     * @brief In each second, this inner loop performs a single pass over the queue
     * to swap adjacent 'B' and 'G' pairs.
     */
    for (int j = 0; j < n; j++) {
      if (a[j] == 'B' && a[j + 1] == 'G') {
        swap(a[j], a[j + 1]);
        /**
         * @brief Skips the next position to ensure each person moves at most once
         * per time step. After swapping, the boy is now at `j+1`, so we can
         * continue the search from `j+2`.
         */
        j += 1;
      }
    }
  }
  cout << a;
}