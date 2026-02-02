/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A - Team.
 *
 * This problem describes a programming contest where a team of three friends
 * (Petya, Vasya, and Tonya) decides whether to implement a problem. They will
 * implement a problem if at least two of them are sure about the solution.
 * The input consists of `n` problems, and for each problem, three integers (0 or 1)
 * indicating whether each friend is sure about the solution. The task is to count
 * how many problems the team will implement.
 *
 * Algorithm: Simple summation and threshold check.
 * For each problem, the code sums the "sureness" values of the three friends.
 * If this sum is 2 or more, it means at least two friends are sure, and a counter
 * for implemented problems is incremented.
 *
 * Time Complexity: O(N), where N is the number of problems, as it involves a single pass
 * through the problems with constant time operations per problem.
 * Space Complexity: O(1) as only a few variables are used to store state.
 */

// the-Quert 231A - Team
#include <iostream> // For standard C++ input/output operations (cin, cout).

using namespace std; // Functional Utility: Brings all identifiers from the std namespace into the current scope, simplifying code by avoiding explicit std:: prefixes.

/**
 * @brief Main function: Solves the Team problem.
 * Functional Utility: Reads the number of problems, then for each problem, reads
 * three integers representing the friends' certainty and counts how many problems
 * the team will solve based on a majority rule.
 * Algorithm: Iterative counting with a simple sum threshold.
 */
int main() {
  int n; // Declares an integer `n` to store the number of problems.
  int Petya, Vasya, Tonya; // Declares integers to store the certainty of each friend for a problem.
  int number(0); // Initializes `number` (count of problems to be implemented) to 0.
  
  cin >> n; // Reads the number of problems `n`.
  
  // Block Logic: Loop `n` times to process each problem.
  // Precondition: `n` is a non-negative integer.
  // Invariant: `number` accurately reflects the count of problems implementable from previous iterations.
  while (n--) {
    cin >> Petya >> Vasya >> Tonya; // Reads the certainty values for Petya, Vasya, and Tonya for the current problem.
    // Functional Utility: Checks if at least two friends are sure about the solution.
    // Logic: If the sum of their certainty (where 1 means sure) is 2 or more, then at least two are sure.
    if (Petya + Vasya + Tonya >= 2) {
      number += 1; // Increment the count of implementable problems.
    }
  }
  
  cout << number << endl; // Prints the total number of problems the team will implement.
  return 0; // Indicates successful program execution.
}