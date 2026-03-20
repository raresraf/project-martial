/**
 * @file 231A.CPP
 * @brief Solution to Codeforces problem 231A - Team.
 *        This program determines how many problems a team will implement,
 *        given that they only implement a problem if at least two out of three
 *        friends are sure about the solution.
 * Algorithm:
 *   1. Read an integer 'n', representing the number of problems.
 *   2. Initialize a counter `res` to 0, which will store the number of problems implemented.
 *   3. Loop 'n' times:
 *      a. Read three integers (a, b, c) representing the friends' certainty about the solution (1 for sure, 0 for not sure).
 *      b. Calculate the sum of these three integers.
 *      c. If the sum is greater than 1 (meaning at least two friends are sure), increment `res`.
 *   4. After processing all problems, print the final value of `res`.
 * Time Complexity: O(n), where n is the number of problems, as each problem is processed once.
 * Space Complexity: O(1) as only a few variables are used to store counts and sums.
 */
#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main() {
  /**
   * @brief Main function implementing the solution for the "Team" problem.
   *
   * Reads the number of problems and for each problem, reads three integers
   * representing the certainty of three friends. It counts how many problems
   * will be implemented based on the rule that at least two friends must be sure.
   * @return int Returns 0 upon successful execution.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(NULL); // Untie cin from cout for faster input.
  int n, a, b, c;
  cin >> n; // Read the number of problems.
  int res = 0; // Initialize a counter for implemented problems.
  // Block Logic: Iterates 'n' times to process each problem.
  // Pre-condition: 'n' specifies the total number of problems to process.
  // Invariant: 'res' accumulates the count of problems that meet the implementation criteria.
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c; // Read the certainty values from the three friends.
    // Block Logic: Checks if at least two friends are sure about the solution.
    // A friend's certainty is represented by 1. If a+b+c > 1, at least two friends are sure.
    if (a + b + c > 1)
      res++; // Increment the counter if the problem will be implemented.
  }
  cout << res; // Print the total number of implemented problems.
}
