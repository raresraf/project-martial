/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A - Team.
 *
 * This program determines the number of problems a team can solve in a contest.
 * A problem is considered solved if at least two out of three team members
 * (Petya, Vasya, and Tonya) are confident in its solution.
 *
 * Algorithm: Direct summation of confidence votes. For each problem, the program
 * reads the confidence votes (1 for confident, 0 for not confident) from the
 * three team members, sums them up, and if the sum is 2 or more, increments
 * the count of solved problems.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1), as only a few variables are stored.
 */

#include <iostream> // Required for standard input/output operations (cin, cout).
#include <string>   // Included but not explicitly used in this solution.
using namespace std; // Uses the standard namespace.

int main() {
  /**
   * @brief Main function to execute the Team problem logic.
   *
   * Reads the number of problems 'n'. Then, for each problem, it reads the
   * confidence votes of three team members. It counts how many problems
   * have at least two confident votes and prints this total.
   */
  // Declare 'n' for the number of problems.
  // 'total' to store the count of problems solved.
  // 'a', 'b', 'c' for the confidence votes of the three team members.
  int n, total = 0, a, b, c;

  // Read the total number of problems.
  cin >> n;
  // Block Logic: Iterate through each problem.
  // Invariant: 'i' iterates from 0 to n-1, processing one problem per loop.
  for (int i = 0; i < n; i++) {
    // Read the three confidence votes for the current problem.
    cin >> a;
    cin >> b;
    cin >> c;
    // Check if the sum of confidence votes is 2 or more, indicating at least two members are confident.
    if (a + b + c >= 2) {
      total++; // Increment the count of solved problems.
    }
  }
  // Output the total number of problems solved.
  cout << total;
  return 0; // Indicate successful program execution.
}
