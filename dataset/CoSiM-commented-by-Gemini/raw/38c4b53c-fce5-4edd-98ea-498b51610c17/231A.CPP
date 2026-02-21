/**
 * @file 231A.CPP
 * @brief Counts the number of problems to be solved based on team confidence.
 *
 * This program calculates how many problems a team of three will implement.
 * The team decides to solve a problem only if at least two of the three
 * members are confident about the solution.
 *
 * Algorithm:
 * 1. Read an integer `n`, which is the total number of problems.
 * 2. Initialize a counter `total` to 0. This counter will store the number of
 *    problems the team will implement.
 * 3. Loop `n` times, once for each problem.
 *    a. For each problem, read three integers (`a`, `b`, `c`), where each integer
 *       is either 1 (confident) or 0 (not confident).
 *    b. Calculate the sum of `a`, `b`, and `c`. This sum represents the number
 *       of team members who are confident.
 *    c. If the sum is 2 or greater, it means at least two members are confident.
 *       In this case, increment the `total` counter.
 * 4. After the loop finishes, print the final value of `total`.
 *
 * Time Complexity: O(N), where N is the number of problems, because the program
 * iterates through each problem once.
 * Space Complexity: O(1), as it uses a fixed amount of memory regardless of the
 * number of problems.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, total = 0, a, b, c;

  // Read the total number of problems.
  cin >> n;

  /**
   * @brief Loop through each problem to check team confidence.
   * Pre-condition: `n` is the number of problems.
   * Invariant: `total` is the count of problems that will be solved based on the
   *            problems evaluated so far.
   */
  for (int i = 0; i < n; i++) {
    // Read the confidence level of each of the three team members.
    cin >> a;
    cin >> b;
    cin >> c;
    
    /**
     * @brief Check if at least two members are confident.
     * The sum of their confidence levels (1 or 0) will be >= 2.
     */
    if (a + b + c >= 2) {
      total++;
    }
  }
  // Print the total number of problems to be implemented.
  cout << total;
}
