/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem from Codeforces.
 * @details The problem involves a team of three friends who decide to implement a
 * programming problem only if at least two of them are confident about the solution.
 * This program counts how many problems the team will implement given `n` problems
 * and the confidence of each member for each problem.
 *
 * @algorithm Simple Counting
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter for implementable problems to zero.
 * 3. Loop `n` times, once for each problem.
 * 4. In each iteration, read the three confidence values (0 or 1) for the team members.
 * 5. Sum the three values. If the sum is 2 or greater, it means at least two members
 *    are confident. In this case, increment the counter.
 * 6. After the loop, print the final value of the counter.
 *
 * @complexity
 * - Time: O(N), where N is the number of problems. The program has a single loop
 *   that runs N times with constant time operations inside.
 * - Space: O(1), for storing a few integer variables.
 */

#include <iostream>
using namespace std;

int main() {
  // n: the total number of problems.
  int n;
  // count: the number of problems the team will implement.
  int count = 0;
  // f1, f2, f3: confidence levels (1 for sure, 0 for not sure) of the three friends.
  int f1, f2, f3;
  
  // Read the number of problems.
  cin >> n;

  /**
   * @brief Iterate through each problem to see if the team will solve it.
   */
  // Invariant: `count` holds the number of solvable problems from the first `i` problems.
  for (int i = 0; i < n; i++) {
    // Read the confidence levels for the current problem.
    cin >> f1 >> f2 >> f3;

    /**
     * @brief Check if at least two members are confident.
     * Summing the three inputs (which are either 0 or 1) is a concise way
     * to count how many members are confident.
     */
    // Pre-condition: The sum must be 2 or more for the team to proceed.
    if (f1 + f2 + f3 >= 2) {
      count++;
    }
  }
  
  // Print the total count of problems the team will implement.
  cout << count << endl;
  
  return 0; // Indicate successful execution.
}
