/**
 * @file 231A.CPP
 * @brief This program solves the "Team" problem. It counts the number of problems
 * a team will solve, given their confidence levels. For each problem, three friends
 * state their confidence (0 or 1). A problem is solved if at least two friends are
 * sure (confidence level of 1).
 * Algorithm: Iterative processing of problems, checking combinations of confidence levels.
 * Time Complexity: O(N), where N is the number of problems, due to a single pass.
 * Space Complexity: O(1) as fixed-size arrays are used, independent of N for processing.
 */
#include <cmath>    // Included in original, though not explicitly used in this solution.
#include <iostream> // Required for input/output operations (cin, cout).
using namespace std;

int main() {
  // Declares three C-style arrays to store confidence levels for up to 1000 problems.
  int num1[1000];
  int num2[1000];
  int num3[1000];
  int counter = 0; // Counts the number of problems the team will solve.
  int y;           // Stores the total number of problems to be considered.
  int i;           // Loop counter.

  // Reads the total number of problems 'y'.
  cin >> y;

  /**
   * Block Logic: Processes each of the 'y' problems.
   * For each problem, it reads the confidence levels of three friends and determines
   * if the problem is solved based on whether at least two friends are confident.
   * Invariant: 'counter' correctly reflects the number of solved problems up to the current iteration.
   */
  for (i = 0; i < y; i++) {
    // Reads the three confidence levels (num1[i], num2[i], num3[i]) for the current problem.
    cin >> num1[i] >> num2[i] >> num3[i];

    /**
     * Conditional Logic: Checks various combinations to determine if at least two friends are confident (value 1).
     * If any of these conditions are met, 'counter' is incremented.
     * This logic covers cases where three friends are confident, or any pair of two friends is confident.
     */
    if (num1[i] == num2[i] && num1[i] == num3[i] && num1[i] == 1) // All three are confident.
      counter++;
    else if (num1[i] == num2[i] && num1[i] == 1) // First two are confident.
      counter++;
    else if (num2[i] == num3[i] && num2[i] == 1) // Second and third are confident.
      counter++;
    else if (num1[i] == num3[i] && num1[i] == 1) // First and third are confident.
      counter++;
  }
  // Prints the total number of problems the team will solve.
  cout << counter << endl;
  // Standard program exit.
  return 0;
}
