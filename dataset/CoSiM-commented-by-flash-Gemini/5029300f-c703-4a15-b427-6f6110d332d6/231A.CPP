/**
 * @file 231A.CPP
 * @brief Counts the number of problems a team will implement, given judge votes.
 *        A problem is implemented if at least two out of three judges agree (vote '1').
 *
 * Problem Name: Young Physicist (Codeforces 231A)
 * Algorithm: Iterative summation of judge votes and conditional counting. For each problem, it sums the three votes
 *            and increments a counter if the sum is 2 or more.
 * Time Complexity: O(N) where N is the number of problems, due to a single linear pass.
 * Space Complexity: O(1) as it uses a fixed number of variables.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, e.g., <iostream>.
using namespace std;

/**
 * @brief Main function to calculate the number of problems to be implemented.
 * @pre Input consists of an integer 'n' (total problems), followed by 'n' lines,
 *      each containing three binary integers (a, b, c) representing judge votes (0 or 1).
 * @post Prints the total count of problems that will be implemented.
 */
int main() {
  int n, a, b, c, ans = 0; // 'n': total problems, 'a, b, c': judge votes, 'ans': total implemented problems.

  // Functional Utility: Reads the total number of problems.
  cin >> n;

  /**
   * Block Logic: Iterates through each problem to read judge votes and determine implementation.
   * Invariant: `ans` accumulates the count of problems where the sum of votes (a + b + c) is 2 or more.
   */
  for (int i = 0; i < n; i++) {
    // Functional Utility: Reads the three judge votes for the current problem.
    cin >> a >> b >> c;

    /**
     * Block Logic: Checks if the current problem will be implemented.
     * Pre-condition: `a`, `b`, `c` hold the binary votes for the current problem.
     * Invariant: If the sum of votes is 2 or more, it means at least two judges agreed.
     */
    if (a + b + c >= 2) { // Checks if at least two judges agreed.
      ans++;              // Increment the count of implemented problems.
    }
  }

  // Functional Utility: Prints the final count of problems to be implemented.
  cout << ans << endl; // Prints the result followed by a newline.
  // No explicit return 0; is needed for main in C++ if it reaches the end.
  // However, for competitive programming, it's often good practice to include it.
}
