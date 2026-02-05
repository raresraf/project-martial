/**
 * @file 231A.CPP
 * @brief Solution to a team programming problem.
 * 
 * This program calculates the number of problems that a three-person team will
 * decide to implement. The team agrees to solve a problem only if at least two
 * of its members are confident about the solution.
 *
 * Algorithm: The program iterates through a given number of problems. For each
 * problem, it sums up the confidence levels (represented as 1 for confident,
 * 0 otherwise) of the three team members. If the sum is 2 or greater, a counter
 * for solved problems is incremented.
 *
 * Time Complexity: O(T), where T is the total number of problems, as the program
 * processes each problem in a single pass with constant time operations inside the loop.
 * Space Complexity: O(1), as a fixed amount of memory is used regardless of the
 * number of problems.
 */
#include <iostream>

using namespace std;

int main() {
  // t: The total number of problems to consider.
  int t;
  cin >> t;

  // x, y, z: Confidence levels of the three team members for a single problem.
  int x, y, z;

  // count: Accumulator for the number of problems the team will implement.
  int count = 0;

  /**
   * Block Logic: Iterates through each problem to decide if it will be solved.
   * Pre-condition: 't' contains the number of problems to process.
   * Invariant: At the beginning of each iteration, 'count' holds the number
   * of problems solved from the previous iterations.
   */
  while (t--) {
    // Read the confidence levels for the current problem.
    cin >> x >> y >> z;

    /**
     * Block Logic: Checks if at least two members are confident.
     * The sum of confidence levels (0 or 1) will be >= 2 if and only if
     * at least two members are confident.
     */
    if ((x + y + z) >= 2) {
      count++;
    }
  }

  // Output the total number of problems that will be implemented.
  cout << count << endl;

  return 0;
}