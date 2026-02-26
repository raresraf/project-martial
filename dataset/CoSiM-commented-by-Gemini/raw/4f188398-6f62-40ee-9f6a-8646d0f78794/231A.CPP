/**
 * @file 231A.CPP
 * @brief A verbose solution to the "Team" problem from Codeforces.
 * @details This program counts how many problems a three-person team will solve,
 * with a problem being solved if at least two members agree.
 *
 * @algorithm
 * 1. Read the number of problems, `n`.
 * 2. Loop `n` times, once for each problem.
 * 3. Inside the loop, use a temporary counter, `current_votes`, initialized to 0.
 * 4. Read the three members' opinions.
 * 5. For each of the three opinions, check if it is '1'. If so, increment
 *    `current_votes`. (This is a verbose way of summing the inputs).
 * 6. After checking all three, if `current_votes` is 2 or greater, increment a
 *    final result counter.
 * 7. Reset `current_votes` to 0 and proceed to the next problem.
 * 8. After the loop, print the final result.
 *
 * @complexity
 * Time Complexity: O(N), for a single pass through the `N` problems.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int i, n;
  int current_votes = 0; // Counter for votes on the current problem.
  int result_count = 0;  // Counter for total problems to be solved.
  int p, v, t;         // Opinions of the three members.

  cin >> n;

  /**
   * @block
   * @brief Loop through each of the `n` problems.
   */
  for (i = 0; i < n; i++) {
    cin >> p >> v >> t;

    /**
     * @block
     * @brief Count the number of '1's for the current problem.
     * @note A more direct way to do this would be `current_votes = p + v + t;`.
     */
    if (p == 1) {
      current_votes++;
    }
    if (v == 1) {
      current_votes++;
    }
    if (t == 1) {
      current_votes++;
    }

    // If two or more members agree, increment the result count.
    if (current_votes >= 2) {
      result_count++;
    }

    // Reset the temporary counter for the next problem.
    current_votes = 0;
  }

  // Print the final total.
  cout << result_count;
  return 0;
}