//
// Created by Taewoo Kim on 3/21/2017.
//

/**
 * @file 231A.CPP
 * @brief Solution for the Codeforces "Team" problem (231A).
 * @details This program counts how many problems a three-person team will attempt.
 * The condition for attempting a problem is that at least two of the three members
 * are confident in the solution. This implementation reads each opinion into a boolean,
 * leveraging C++'s implicit conversion of 0 to false and non-zero to true.
 *
 * @algorithm Iterative Counting
 * @time_complexity O(n), where 'n' is the number of problems. The nested loops
 * result in 3*n iterations, which is linear.
 * @space_complexity O(1), as memory usage is constant.
 */
#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;

  int solvable = 0; // Counter for problems that will be solved.
  
  /**
   * Block Logic: Iterates through each of the 'n' problems.
   */
  for (int p = 0; p < n; p++) {
    int total = 0; // Tally of confident members for the current problem.
    
    /**
     * Block Logic: Gathers the opinions of the three team members.
     */
    for (int i = 0; i < 3; i++) {
      bool opinion;
      // When an integer is read into a bool, 0 becomes false (0) and any
      // non-zero value becomes true (1). This is used to sum the 'yes' votes.
      cin >> opinion;
      total += opinion;
    }
    
    // Pre-condition: Check if the number of confident members is 2 or more.
    if (total >= 2)
      ++solvable;
  }

  // Output the final count of solvable problems.
  cout << solvable;

  return 0;
}
