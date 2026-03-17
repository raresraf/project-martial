/**
 * @file 231A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 231A "Team".
 *
 * @details
 * This program calculates the number of problems a team of three will solve.
 * The team decides to solve a problem only if at least two of its members are
 * confident about the solution. The input consists of `n` problems, and for
 * each problem, three boolean values (0 or 1) indicate the confidence of each
 * member.
 *
 * @algorithm
 * The algorithm iterates `n` times, once for each problem. In each iteration,
 * it reads the three confidence values. By summing these values (which are
 * either 0 or 1), it can quickly determine how many members are confident. If
 * the sum is 2 or more, a counter for solvable problems is incremented.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the number of problems, as the program
 *   loops through the input once.
 * - Space Complexity: O(1), as it only uses a fixed number of variables
 *   for storage regardless of the input size.
 */

#include <iostream>

using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  int n; // The total number of problems.
  int k, m, in; // Variables to store the confidence of the three members for each problem (0 or 1).
  int s = 0; // Counter for the number of problems the team will solve.
  
  // Read the number of problems.
  cin >> n;
  
  /**
   * @brief Iterates through each problem to determine if it will be solved.
   */
  for (int i = 0; i < n; i++) {
    // Read the confidence levels for the current problem.
    cin >> k >> m >> in;
    
    /**
     * @brief Check if at least two members are confident.
     * Since confident members are represented by 1 and non-confident by 0,
     * the sum of the three inputs gives the number of confident members.
     * If this sum is greater than 1 (i.e., 2 or 3), the team solves the problem.
     */
    if (k + m + in > 1) {
      s++;
    }
  }
  
  // Print the total count of solvable problems.
  cout << s << endl;
  return 0;
}
