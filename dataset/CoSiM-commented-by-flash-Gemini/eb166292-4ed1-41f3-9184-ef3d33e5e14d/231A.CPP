//#include <bits/stdc++> // Common competitive programming include, commented out here.
#include <iostream>

/**
 * @file 231A.CPP
 * @brief Solution to the "Team" competitive programming problem (Codeforces 231A).
 * This program determines how many problems a team can implement. For each problem,
 * three friends indicate if they are sure about the solution (1) or not (0).
 * A problem is implemented if at least two friends are sure about it.
 * Algorithm: Iterative counting of problems where the sum of sure votes is greater than 1.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1), as only a few scalar variables are used.
 */
int main() {
  // Declares 'total' for the number of problems.
  int total = 0;
  // Declares 'a', 'b', 'c' for individual friend's opinions (presumably binary: 0 or 1).
  int a, b, c;
  // Initializes 'a', 'b', 'c' to 0. (These initializations are redundant as they are overwritten by cin).
  a = 0;
  b = 0;
  c = 0;
  // Declares 'j' to store the sum of opinions for a single problem.
  int j = 0;
  // Declares 'total_to_use' to count the number of implementable problems.
  int total_to_use = 0;
  
  // Reads the total number of problems from standard input.
  std::cin >> total;
  
  /**
   * Block Logic: Iterates 'total' times, once for each problem.
   * Invariant: At the start of each iteration, 'i' is the current problem index (0-based).
   * Pre-condition: 'total' is a positive integer indicating the number of problems to follow.
   * Post-condition: 'total_to_use' holds the total count of implementable problems.
   */
  for (int i = 0; i < total; i++) {
    // Reads the opinions of the three friends for the current problem.
    // Each opinion (a, b, c) is expected to be 0 or 1.
    std::cin >> a;
    std::cin >> b;
    std::cin >> c;
    
    // Calculates the sum of opinions for the current problem.
    j += a + b + c;
    
    /**
     * Conditional Logic: Checks if at least two friends are sure about the solution.
     * If 'a', 'b', 'c' are 0 or 1, their sum 'j' will be greater than 1 if and only if
     * at least two of them are 1 (i.e., at least two friends are sure).
     */
    if (j > 1)
      total_to_use++; // Increments the counter for implementable problems.
    
    // Resets 'j', 'a', 'b', 'c' for the next iteration (j reset is necessary, others are overwritten).
    j = 0;
    a = 0;
    b = 0;
    c = 0;
  }
  // Prints the final count of implementable problems.
  std::cout << total_to_use;
  
  return (0); // Standard practice to indicate successful execution.
}
