/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces (Problem 231A).
 *        This program determines the number of problems a team can implement.
 *        For each problem, three friends express their opinion (1 if sure, 0 if not).
 *        A problem is implemented if at least two friends are sure about the solution.
 * Algorithm: Simple Counting / Conditional Logic
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */

#include <cstdio> // Required for input/output operations (scanf, printf).
using namespace std;

/**
 * @brief Main function to read problem opinions and count solvable problems.
 * @return 0 if the program executes successfully.
 */
int main() {
  // Declare variables:
  // n: total number of problems.
  // a, b, c: opinions of the three friends for a single problem (1 for sure, 0 for not).
  // x: counter for the number of problems that can be implemented.
  int n, a, b, c, x = 0;
  // Read the total number of problems from standard input.
  scanf("%d", &n);
  
  /**
   * Block Logic: Iterates through each problem to determine if it can be implemented.
   * Invariant: 'n' decrements after each problem, 'x' accumulates the count of solvable problems.
   */
  while (n--) { // Loop 'n' times, once for each problem.
    // Reset opinions for the current problem.
    a = 0;
    b = 0;
    c = 0;
    // Read the opinions of the three friends for the current problem.
    scanf("%d%d%d", &a, &b, &c);
    
    // Block Logic: Check if at least two friends are sure about the solution.
    // Pre-condition: 'a', 'b', 'c' are either 0 or 1.
    // Post-condition: 'x' is incremented if the sum of opinions is 2 or more.
    if (a + b + c >= 2)
      x++; // If the sum of opinions is 2 or more, increment the solvable problems counter.
  }
  // Print the total number of problems that can be implemented.
  printf("%d", x);
  return 0; // Indicate successful program termination.
}
