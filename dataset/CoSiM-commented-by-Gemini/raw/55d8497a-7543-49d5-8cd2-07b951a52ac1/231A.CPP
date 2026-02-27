/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 * @details The problem counts how many out of `n` problems a team of three will
 * solve. A problem is solved if at least two team members are confident in the
 * solution.
 *
 * @algorithm
 * 1. Read the number of problems, `n`.
 * 2. Initialize a `solutionCount` to 0.
 * 3. Loop `n` times for each problem.
 *    a. Read the three members' confidence values (0 or 1) into an array.
 *    b. Initialize a temporary counter `sureCount` to 0.
 *    c. Iterate through the array of confidence values. For each '1',
 *       increment `sureCount`.
 *    d. If `sureCount` is 2 or more, increment `solutionCount`.
 * 4. After the loop, print the final `solutionCount`.
 *
 * @complexity
 * Time Complexity: O(n), as for each of the `n` problems, a constant number
 * of operations are performed.
 * Space Complexity: O(1), as the space used is constant regardless of `n`.
 */
#include <iostream>

using std::cin;
using std::cout;

int main(void) {
  int n; // The total number of problems.
  // An array to hold the confidence values (0 or 1) for a single problem.
  int sures[] = {0, 0, 0};

  cin >> n;
  
  // `count` is used as the loop counter for the `while` loop.
  int count = 0;
  // `solutionCount` is the final answer, a tally of problems to be solved.
  int solutionCount = 0;

  /**
   * @block
   * @brief Main loop to process each of the `n` problems.
   */
  while (count++ < n) {
    cin >> sures[0] >> sures[1] >> sures[2];
    
    int sureCount = 0; // Temporary counter for the current problem.
    
    /**
     * @block
     * @brief Count how many members are sure about the current problem.
     */
    for (int i = 0; i < 3; i++) {
      if (sures[i]) { // In C++, non-zero integers evaluate to true.
        sureCount++;
      }
    }
    
    // If two or more members are sure, increment the total solution count.
    if (sureCount >= 2) {
      solutionCount++;
    }
  }

  cout << solutionCount;

  return 0;
}
