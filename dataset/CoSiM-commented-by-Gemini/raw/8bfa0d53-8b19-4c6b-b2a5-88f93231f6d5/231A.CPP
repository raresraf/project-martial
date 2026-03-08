/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem on Codeforces.
 *
 * @details
 * This program counts the number of problems a three-person team will solve.
 * The condition for solving a problem is that at least two members are sure
 * about the solution. The program reads the number of problems, then iterates
 * through each one, summing the confidence votes (1 for sure, 0 for not) to
 * see if the total is 2 or more.
 *
 * @section
 * Algorithm: Iterative counting via summation.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;

// Global variables to hold state. While functional for small scripts, this
// is not a recommended practice in larger applications.
int a, b, c, d, sum = 0; // a: problem count, b,c,d: votes, sum: result

int main() {
  // Read the total number of problems into global variable 'a'.
  cin >> a;

  /**
   * @brief This loop iterates through each of the 'a' problems.
   */
  for (int i = 0; i < a; i++) {
    // Read the three confidence votes for the current problem.
    cin >> b >> c >> d;
    /**
     * @brief Check if the team will implement the problem.
     *
     * Logic: Since confidence is represented as 1 (sure) or 0 (not sure),
     * summing the three votes and checking if the result is greater than 1
     * is a concise way to see if at least two members are confident.
     */
    if ((b + c + d) > 1) {
      sum++; // Increment the total count of solvable problems.
    }
  }
  // Output the final count.
  cout << sum;
}
