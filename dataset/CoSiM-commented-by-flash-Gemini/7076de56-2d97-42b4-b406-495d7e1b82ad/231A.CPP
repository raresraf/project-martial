/**
 * @file 231A.CPP
 * @brief Solution to Codeforces Problem 231A: "Team".
 * Determines the number of problems a team can solve. A problem is solvable if
 * at least two out of three friends are sure about the solution.
 * Algorithm: For each problem, count the number of friends who are sure. If this count
 * is greater than one, increment a total solved problems counter.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1) as only a few variables are used.
 */
#include <iostream> // Required for input/output operations (cin, cout).
using namespace std;

/**
 * @brief Main function to solve the "Team" problem.
 * Reads the number of problems, then for each problem, reads the certainty of three friends,
 * and calculates the total number of problems that can be solved.
 */
int main() {
  // Declare variables:
  // 'i': loop counter.
  // 'n': total number of problems.
  // 'a', 'b', 'c': certainty of three friends for a single problem (1 for sure, 0 for not sure).
  // 'ct': total count of problems that can be solved.
  // 'ct1': count of friends sure about the current problem's solution.
  int i, n, a, b, c, ct = 0, ct1;
  cin >> n; // Reads the total number of problems.

  /**
   * Block Logic: Iterates through each problem to determine if it can be solved.
   * Pre-condition: 'n' specifies the number of problems to evaluate.
   * Invariant: After each iteration, one problem has been evaluated, and 'ct'
   *            accurately reflects the number of solvable problems found so far.
   */
  for (i = 1; i <= n; i++) {
    cin >> a >> b >> c; // Reads the certainty (0 or 1) of the three friends for the current problem.
    ct1 = 0;             // Reset the counter for friends sure about the current problem.
    
    // Check each friend's certainty and increment 'ct1' if they are sure.
    if (a == 1)
      ct1++;
    if (b == 1)
      ct1++;
    if (c == 1)
      ct1++;
    
    /**
     * Block Logic: Checks if at least two friends are sure about the solution.
     * If 'ct1' is greater than 1, the problem can be solved.
     */
    if (ct1 > 1)
      ct++; // Increment the total count of solvable problems.
  }
  cout << ct; // Prints the final count of problems that can be solved.
  return 0;
}
