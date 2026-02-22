/**
 * @file 231A.CPP
 * @brief A solution to the "Team" problem (Codeforces 231A).
 *
 * @algorithm
 * The problem is to count the number of problems that a three-person team will
 * decide to implement. The condition for implementation is that at least two of
 * the three members are confident about the solution.
 *
 * The algorithm iterates through each problem and checks the condition directly:
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter for solvable problems.
 * 3. Loop `n` times.
 * 4. For each problem, read the three confidence values (0 or 1).
 * 5. Use a series of `if-else if` statements to check for all possible
 *    combinations where at least two members are confident.
 * 6. If any of these conditions are met, increment the counter.
 * 7. After the loop, print the final count.
 *
 * @complexity
 * - Time Complexity: O(N), for a single pass through the N problems.
 * - Space Complexity: O(1), as storage is constant (only for one problem's
 *   worth of data at a time).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, problem = 0;
  cin >> n;
  int arr[3];

  /**
   * @brief Main loop to process each of the `n` problems.
   */
  while (n > 0) {
    // Read the three opinions for the current problem.
    for (int i = 0; i < 3; i++)
      cin >> arr[i];

    /**
     * @brief A verbose and partially redundant chain of checks to see if at
     * least two members are confident. A simpler condition would be
     * `if (arr[0] + arr[1] + arr[2] >= 2)`.
     * The final `else if` for all three being 1 is completely redundant, as
     * that case would be caught by the first `if`.
     */
    if (arr[0] == 1 && arr[1] == 1)
      problem++;
    else if (arr[1] == 1 && arr[2] == 1)
      problem++;
    else if (arr[0] == 1 && arr[2] == 1)
      problem++;
    else if (arr[0] == 1 && arr[1] == 1 && arr[2] == 1) // Redundant check
      problem++;
    n--;
  }
  cout << problem;
}
