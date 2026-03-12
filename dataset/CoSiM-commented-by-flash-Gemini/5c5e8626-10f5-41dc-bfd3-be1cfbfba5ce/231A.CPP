/**
 * @5c5e8626-10f5-41dc-bfd3-be1cfbfba5ce/231A.CPP
 * @brief This program solves the "Team" problem (Codeforces 231A). It reads 'n' problems,
 * and for each problem, three binary inputs (0 or 1) representing whether each of three
 * friends is sure about the solution. A problem is counted if at least two friends are
 * sure about its solution.
 *
 * Algorithm: Iterative processing of problems, checking the sum of binary inputs for a majority.
 * Time Complexity: O(N) where N is the number of problems.
 * Space Complexity: O(1) for storing counters and input.
 */

#include <iostream> // Includes the standard input/output stream library.
using namespace std; // Uses the standard namespace.

int main() {
  int n, count = 0, f1, f2, f3; // Declares 'n' for the number of problems, 'count' (solvable problems) initialized to 0,
                               // and 'f1', 'f2', 'f3' for each friend's certainty.
  cin >> n; // Reads the number of problems from standard input.
  
  /**
   * Block Logic: Processes 'n' problems. For each problem, it reads the certainty of three friends
   * and increments a counter if at least two friends are sure.
   * Invariant: 'count' accurately reflects the cumulative number of implemented problems after each iteration.
   */
  for (int i = 0; i < n; i++) { // Loops 'n' times, once for each problem.
    cin >> f1 >> f2 >> f3; // Reads the three binary certainty values (0 or 1) for the current problem.
    // Inline: Checks if the sum of the three certainty values is 2 or more.
    // Since inputs are 0 or 1, a sum of 2 or 3 implies at least two friends are sure.
    if (f1 + f2 + f3 >= 2) 
      count++; // Increments the total count of implemented problems.
  }
  cout << count << endl; // Prints the final number of problems the team will solve, followed by a newline.
  return 0; // Indicates successful program termination.
}
