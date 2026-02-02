/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A - Team.
 *
 * This problem involves a team of three members deciding to solve a problem if
 * at least two of them are sure about the solution. For each of `n` problems,
 * three integers (0 or 1) represent whether each member is sure. The task is
 * to count the total number of problems the team will implement.
 *
 * Algorithm: Conditional counting with pairwise checks.
 * The code iterates through each problem. For each problem, it checks if any
 * two of the three members (a, b, c) are simultaneously '1' (sure). If any
 * such pair exists, the problem count is incremented.
 *
 * Time Complexity: O(N), where N is the number of problems, as it processes
 * each problem with a constant number of comparisons.
 * Space Complexity: O(1) as only a few variables are used to store state.
 */

#include <bits/stdc++.h> // Includes most standard C++ libraries, common in competitive programming.
using namespace std; // Functional Utility: Brings all identifiers from the std namespace into the current scope.

/**
 * @brief Main function: Solves the Team problem.
 * Functional Utility: Reads the number of problems `n`, then for each problem,
 * reads three integers representing whether each of three team members is sure
 * about the solution. It counts and prints the number of problems where at
 * least two members are sure.
 * Algorithm: Iterative counting based on pairwise agreement.
 */
int main() {
  int n, cn = 0; // `n`: number of problems; `cn`: count of problems the team will solve.
  scanf("%d", &n); // Reads the number of problems using `scanf`.
  int i; // Loop counter.
  
  // Block Logic: Loop `n` times to process each problem.
  // Precondition: `n` is a non-negative integer.
  // Invariant: `cn` correctly counts the solvable problems from previous iterations.
  for (i = 0; i < n; i++) {
    int a, b, c; // Integers representing the certainty of each of the three team members (0 for unsure, 1 for sure).
    scanf("%d %d %d", &a, &b, &c); // Reads the certainty values for the current problem.
    // Functional Utility: Check if at least two members are sure about the solution.
    // Logic: Explicitly checks all three possible pairs for agreement.
    if (a == 1 && b == 1) // If member A and B are sure.
      cn++;
    else if (b == 1 && c == 1) // Else if member B and C are sure.
      cn++;
    else if (c == 1 && a == 1) // Else if member C and A are sure.
      cn++;
  }
  printf("%d\n", cn); // Prints the total number of problems the team will solve.
  return 0; // Indicate successful program execution.
}