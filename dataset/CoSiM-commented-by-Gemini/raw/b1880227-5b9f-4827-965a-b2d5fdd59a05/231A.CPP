/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem.
 * @details This program calculates the number of problems a team will implement.
 * The team decides to implement a problem if at least two out of its three members
 * are sure about the solution.
 *
 * @algorithm
 * 1. Read the total number of problems `N`.
 * 2. Loop `N` times, once for each problem.
 * 3. Inside the loop, initialize a temporary counter `cnt` to 0.
 * 4. Use a nested loop to read the three members' opinions for the current problem.
 * 5. For each opinion that is '1' (sure), increment `cnt`.
 * 6. After reading the three opinions, if `cnt` is 2 or greater, increment the
 *    final result counter `res`.
 * 7. After the main loop finishes, print the total `res`.
 *
 * @complexity
 * - Time: O(N), as each of the N problems is processed in a constant number of operations (3 reads, 3 checks).
 * - Space: O(1), as the memory usage is constant regardless of the input size.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // N: The total number of problems.
  // cnt: A temporary counter for 'sure' members for a single problem.
  // dm: A temporary variable to store a single member's opinion (1 or 0).
  // res: The final result, counting problems with >= 2 'sure' members.
  int N, cnt = 0, dm, res = 0;
  cin >> N;
  
  /**
   * @brief Main loop iterating through each of the N problems.
   */
  for (int i = 1; i <= N; i++) {
    // Reset the 'sure' count for the new problem.
    cnt = 0;
    /**
     * @brief Inner loop to read 3 opinions and count the 'sure' ones.
     */
    for (int j = 1; j <= 3; j++) {
      cin >> dm;
      if (dm == 1)
        cnt++;
    }
    // Pre-condition: `cnt` holds the number of members sure about the current problem.
    // Post-condition: `res` is incremented if the team agrees to solve it.
    if (cnt >= 2)
      res++;
  }
  
  // Output the total number of problems to be implemented.
  cout << res << endl;
}