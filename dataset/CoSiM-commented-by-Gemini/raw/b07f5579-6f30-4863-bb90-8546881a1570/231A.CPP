/**
 * @file 231A.CPP
 * @brief Solution to a problem requiring counting sets with a minimum number of 'yes' votes.
 * @details This program counts how many out of N sets of three binary opinions have a sum of at least 2.
 * It's a common competitive programming problem about team decisions.
 *
 * @algorithm
 * 1. Reads the total number of problems, `n`.
 * 2. Iterates `n` times, once for each problem.
 * 3. In each iteration, it reads three opinions, sums them up.
 * 4. If the sum is 2 or greater, a counter is incremented.
 * 5. The final count is printed.
 *
 * @complexity
 * - Time: O(N), as it processes each of the N problems in constant time (3 reads and a sum).
 * - Space: O(1), as the memory usage (`arr`, `total`, etc.) is constant regardless of N.
 */
#include <iostream>

using namespace std;

int main() {
  // n: The total number of problems to evaluate.
  // arr: A small array to temporarily store the three opinions for a single problem.
  // count: The final counter for problems that meet the criteria (at least 2 'yes' votes).
  // total: A temporary variable to sum the opinions for one problem.
  int n, arr[1000], count = 0, total = 0;
  cin >> n;

  /**
   * @brief Main loop to process each of the `n` problems.
   * @invariant At the start of iteration `i`, `count` holds the number of valid problems from the first `i` problems processed.
   */
  for (int i = 0; i < n; i++) {
    /**
     * @brief Inner loop to read and sum the 3 opinions for the current problem.
     */
    for (int i = 0; i < 3; i++) {
      cin >> arr[i];
      total += arr[i];
    }

    // Pre-condition: `total` contains the sum of the three opinions for the current problem.
    // Post-condition: `count` is incremented if the sum is 2 or more.
    if (total >= 2)
      count += 1;
    
    // Reset the sum for the next problem.
    total = 0;
  }

  // Output the final count of problems to be implemented.
  cout << count << endl;

  return 0;
}
