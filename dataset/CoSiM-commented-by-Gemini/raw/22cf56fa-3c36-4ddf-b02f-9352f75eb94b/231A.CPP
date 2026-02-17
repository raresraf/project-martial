/**
 * @file 231A.CPP
 * @brief Solution for a counting problem based on a team's consensus.
 * @details This program counts how many problems will be solved out of a total
 * number of submissions. A problem is considered solved only if at least two out
 * of three team members are confident in the solution.
 *
 * @section
 *
 * Algorithm:
 * The program first reads the total number of problems. It then enters a loop
 * that iterates through each problem. In each iteration, it reads the confidence
 * levels of the three members (as 1 for confident, 0 for not). By summing these
 * three values, it checks if the total is two or greater. If this condition is
 * met, a counter for solved problems is incremented. Finally, the total count
 * is printed.
 *
 * - Time Complexity: O(N), where N is the number of problems.
 * - Space Complexity: O(1).
 */
#include <iostream>
using namespace std;
int main() {
  int i, j, c = 0; // i: total problems, j: loop counter, c: solved problem count
  cin >> i;
  /**
   * Block Logic: Iterates 'i' times, once for each problem.
   * Invariant: Before each loop, 'c' holds the count of problems solved so far.
   */
  for (j = 0; j < i; j++) {
    int p, q, r; // p, q, r: confidence levels of the three members
    cin >> p >> q >> r;
    /**
     * Block Logic: Checks if the consensus threshold (>=2) is met.
     * Pre-condition: p, q, and r are either 0 or 1.
     */
    if ((p + q + r) >= 2)
      c++;
  }
  cout << c << endl;
  return 0;
}
