/**
 * @file 231A.CPP
 * @brief This program solves the "Team" problem. It counts the number of problems
 * a team will solve, given their confidence levels. For each problem, three friends
 * state their confidence (0 or 1). A problem is solved if at least two friends are
 * sure (confidence level of 1).
 * Algorithm: Reads confidence levels into a 2D array and then iterates to check
 * if the sum of confidence for each problem is at least 2.
 * Time Complexity: O(N * C), where N is the number of problems and C is the fixed number of friends (3).
 * Space Complexity: O(N * C) for storing confidence levels in the 2D array.
 */
#include <iostream> // Required for input/output operations (std::cin, std::cout).
#include <string>   // Included in original, though not explicitly used for this solution.
#define ll long long    // Defines a macro 'll' as a shorthand for 'long long'.
#define ld long double  // Defines a macro 'ld' as a shorthand for 'long double', not used here.

int main() {
  int n;      // Stores the total number of problems to be considered.
  int output = 0; // Counts the number of problems the team will solve.
  // Reads the total number of problems 'n'.
  std::cin >> n;
  // Declares a Variable Length Array (VLA) to store confidence levels.
  // 'a[n][3]' means 'n' problems, each with 3 confidence levels.
  // Note: VLAs are a C99 feature and a GNU extension in C++, not standard C++.
  int a[n][3];

  /**
   * Block Logic: Reads the confidence levels for each problem.
   * For each of 'n' problems, it reads 3 integer values representing the confidence
   * of each friend (0 or 1) and stores them in the 2D array 'a'.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      std::cin >> a[i][j];
    }
  }

  /**
   * Block Logic: Iterates through each problem to determine if it's solved.
   * A problem is considered solved if at least two out of three friends have a
   * confidence level of 1. The `output` counter is incremented for each solved problem.
   * This implementation checks for confidence by explicitly evaluating various combinations.
   * Invariant: `output` correctly reflects the number of solved problems up to the current iteration.
   */
  for (int i = 0; i < n; i++) {
    // Conditional Logic: Checks if at least two friends are confident for the current problem.
    // The problem is solved if the sum of confidence levels (0 or 1) for the three friends is 2 or more.
    // This implementation explicitly checks combinations rather than summing.
    // Checks if the first two friends are confident (both 1).
    if (a[i][0] == 1 && a[i][0] == a[i][1]) {
      output++;     // Increment solved problems counter.
      continue;     // Move to the next problem.
    }
    // Checks if the second and third friends are confident (both 1).
    else if (a[i][1] == 1 && a[i][1] == a[i][2]) {
      output++;     // Increment solved problems counter.
      continue;     // Move to the next problem.
    }
    // Checks if the first and third friends are confident (both 1).
    else if (a[i][0] == 1 && a[i][0] == a[i][2]) {
      output++;     // Increment solved problems counter.
      continue;     // Move to the next problem.
    }
  }

  // Prints the total number of problems the team will solve.
  std::cout << output;

  // Standard program exit.
  return 0;
}