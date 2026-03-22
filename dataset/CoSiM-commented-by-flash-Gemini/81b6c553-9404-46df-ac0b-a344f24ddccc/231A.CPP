/**
 * @file 231A.CPP
 * @brief Solution to Codeforces problem 231A - Team.
 *        This program determines how many problems a team of three friends will implement
 *        in a programming contest. A problem is implemented if at least two out of the three
 *        friends are sure about its solution.
 *
 * Algorithm:
 *   1. Reads an integer `n`, representing the total number of problems to evaluate.
 *   2. Initializes `co` to 0, which will count the problems the team decides to implement.
 *   3. Iterates `n` times, once for each problem:
 *      a. Reads three integers (`a`, `b`, `c`), each representing a friend's confidence
 *         (1 if sure, 0 if not).
 *      b. If the sum of their confidence levels (`a + b + c`) is 2 or greater,
 *         it indicates at least two friends are sure, so `co` is incremented.
 *   4. After processing all problems, prints the final `co`.
 *
 * Time Complexity: O(n), where `n` is the number of problems, due to the single loop.
 * Space Complexity: O(1) for storing a few integer variables.
 */

// coding team codeforces

#include <bits/stdc++.h> // Includes common C++ standard library headers.
using namespace std;

int main() {
  /**
   * @brief Main function that solves the "Team" problem.
   *
   * Reads the number of problems and the confidence levels of three friends
   * for each problem, then counts how many problems they will attempt.
   *
   * @return int Returns 0 upon successful execution.
   */
  int n; // Total number of problems to be evaluated.
  cin >> n; // Reads the number of problems.
  
  int a, b, c; // Confidence levels for each friend (1 for sure, 0 for not).
  int co = 0;  // Counter for problems the team will implement.
  
  /**
   * Block Logic: Iterates through each problem, processing the friends' confidence levels.
   * Invariant: `co` accurately reflects the count of problems implemented so far.
   * Pre-condition: `n` is a non-negative integer.
   */
  while (n--) { // Loop continues for `n` iterations, decrementing `n` each time.
    cin >> a >> b >> c; // Reads confidence levels for the current problem.
    /**
     * Conditional Logic: Checks if at least two friends are sure about the current problem.
     * The sum of confidence levels (0 or 1) will be >= 2 if at least two friends are sure.
     */
    if (a + b + c >= 2)
      co++; // Increments the count of implemented problems.
  }

  // Prints the total number of problems the team will implement.
  cout << co << endl;
  return 0;
}
