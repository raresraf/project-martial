/**
 * @file 231A.CPP
 * @brief Counts the number of problems that will be implemented.
 *
 * This program solves a problem where 'n' programming contest problems are given,
 * and for each problem, three friends decide if they are sure about the solution.
 * A problem is implemented if at least two friends are sure. This program counts
 * how many problems will be implemented.
 *
 * @b Algorithm: Brute-force Counting
 * @b Time_Complexity: O(N), where N is the number of problems, as inner loops are constant time.
 * @b Space_Complexity: O(1)
 */
#include <iostream>

using namespace std;

int main() {
  // 'n' is the total number of problems.
  int n;
  // 'a' stores the three friends' opinions (1 for sure, 0 otherwise).
  int a[5]; // Note: Array is larger than needed (3), but functionally correct.
  // 'flag' counts the number of problems that will be implemented.
  int flag = 0;
  cin >> n;

  /**
   * @brief Main loop to iterate through each problem.
   * Pre-condition: 'n' holds the total number of problems.
   * Invariant: 'flag' contains the count of implementable problems from the first 'i' problems.
   */
  for (int i = 0; i < n; i++) {
    // Read the opinions of the three friends for the current problem.
    for (int j = 0; j < 3; j++) {
      cin >> a[j];
    }
    
    // 'count' tracks how many friends are sure about the current problem.
    int count = 0;
    /**
     * @brief Check the opinions for the current problem.
     * Logic: Iterate through the three opinions and count how many are '1'.
     * If the count reaches 2, the problem is implementable, so increment 'flag'
     * and break early since the condition is met.
     */
    for (int j = 0; j < 3; j++) {
      if (a[j] == 1) {
        count++;

        if (count >= 2) {
          flag++;
          break;
        }
      }
    }
  }

  // Output the total number of implementable problems.
  cout << flag;
  return 0;
}
