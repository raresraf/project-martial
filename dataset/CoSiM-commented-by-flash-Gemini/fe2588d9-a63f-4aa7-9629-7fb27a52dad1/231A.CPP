/**
 * @file 231A.CPP
 * @brief Solution to the "Team" competitive programming problem (Codeforces 231A).
 * This program determines how many problems a team can implement. For each problem,
 * three friends indicate if they are sure about the solution (1) or not (0).
 * A problem is implemented if at least two friends are sure about it.
 * Algorithm: Iterative counting of problems where the sum of sure votes is >= 2.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1), as only a few scalar variables are used.
 */
#include <iostream>

int main() {
  // Declares 'n' for the total number of problems, 'a', 'b', 'c' for individual friend's opinions,
  // and 'counter' to keep track of the number of problems that can be implemented.
  int n, a, b, c, counter = 0;
  // Reads the total number of problems 'n'.
  std::cin >> n;
  
  /**
   * Block Logic: Iterates 'n' times, once for each problem.
   * Invariant: At the start of each iteration, 'i' is the current problem index (0-based).
   * Pre-condition: 'n' is a positive integer indicating the number of problems to follow.
   * Post-condition: 'counter' holds the total count of implementable problems.
   */
  for (int i = 0; i < n; i++) {
    // Reads the opinions of the three friends for the current problem.
    // Assuming 'a', 'b', 'c' are binary (0 or 1).
    std::cin >> a;
    std::cin >> b;
    std::cin >> c;

    /**
     * Conditional Logic: Checks if at least two friends are sure about the solution.
     * If 'a', 'b', 'c' are 0 or 1, their sum will be >= 2 if and only if at least two of them are 1.
     */
    if (a + b + c >= 2) {
      counter += 1; // Increments the counter for implementable problems.
    }
  }
  // Prints the final count of implementable problems.
  std::cout << counter << std::endl;
  
  return 0; // Standard practice to indicate successful execution.
}