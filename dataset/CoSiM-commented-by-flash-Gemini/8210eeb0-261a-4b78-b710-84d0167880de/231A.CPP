/**
 * @file 231A.CPP
 * @brief Solution to a competitive programming problem involving counting solvable problems.
 *
 * This program reads a series of problem statements, each consisting of three
 * integer votes (or skill levels). A problem is considered "solvable" if
 * at least two of the three votes are positive (i.e., their sum is 2 or more).
 * The objective is to count the total number of solvable problems.
 *
 * Algorithm: Iterative counting.
 * Time Complexity: O(N) where N is the number of problem statements (line_count).
 * Space Complexity: O(1) as only a few variables are used.
 */

#include <iostream>

/**
 * @brief Standard namespace for C++ functionalities.
 */
using namespace std;

/**
 * @brief Main function where the program execution begins.
 * @return int Returns 0 on successful execution.
 */
int main() {
  int line_count;
  /**
   * @brief Reads the number of problem statements to process.
   *
   * Pre-condition: User provides a valid integer input representing the total number of test cases.
   * Post-condition: `line_count` holds the number of iterations for the subsequent loop.
   */
  cin >> line_count;

  int member1, member2, member3, count = 0;

  /**
   * @brief Processes each problem statement.
   *
   * Invariant: `line_count` decrements with each iteration, and `count` accumulates
   *            the number of solvable problems found so far.
   * Pre-condition: `line_count` contains the number of problems to evaluate.
   * Post-condition: `count` holds the final tally of solvable problems.
   */
  while (line_count--) {
    /**
     * @brief Reads three integer values for the current problem statement.
     *
     * Pre-condition: Three integer values are available in the input stream.
     * Post-condition: `member1`, `member2`, `member3` hold the respective input values.
     */
    cin >> member1 >> member2 >> member3;
    /**
     * @brief Checks if the current problem is solvable.
     *
     * Conditional Logic: A problem is solvable if the sum of its three member scores
     *                    is greater than or equal to 2.
     * Functional Utility: Increments `count` if the solvability condition is met.
     */
    if (member1 + member2 + member3 >= 2)
      count++;
  }
  /**
   * @brief Prints the total number of solvable problems.
   *
   * Pre-condition: `count` holds the accurate total of solvable problems.
   * Post-condition: The computed `count` is displayed on the console.
   */
  cout << count;
  return 0;
}