/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem (Codeforces 231A).
 *
 * @algorithm
 * The problem requires counting the number of problems for which at least two
 * out of three team members are confident in the solution. The input for each
 * problem consists of three integers (0 or 1), where 1 signifies confidence.
 *
 * The algorithm is as follows:
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter for solvable problems to zero.
 * 3. Loop `n` times, once for each problem.
 * 4. In each iteration, read the three members' confidence values.
 * 5. Sum these three values. If the sum is 2 or greater, it means at least
 *    two members are confident.
 * 6. If the condition in step 5 is met, increment the counter.
 * 7. After the loop finishes, print the final count.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the number of problems, as we process
 *   each problem in constant time.
 * - Space Complexity: O(1), as the storage required (for the three opinions)
 *   does not depend on N.
 */
#include <iostream>
using namespace std;
int main() {
  int n;          // Total number of problems.
  int count1 = 0; // Counter for the number of problems that will be implemented.
  int a[3];       // Array to store the three members' opinions for a single problem.

  // Read the number of problems.
  cin >> n;
  /**
   * @brief Main loop to process each problem.
   * Pre-condition: `n` contains the total number of problems.
   * Invariant: `count1` holds the number of solvable problems found so far among
   * the first `i` problems.
   */
  for (int i = 0; i < n; i++) {
    /**
     * @brief Reads the opinions of the three team members for the current problem.
     */
    for (int j = 0; j < 3; j++) {
      cin >> a[j];
    }
    /**
     * @brief Checks if at least two members are confident.
     * The condition is verbose but correct. A simpler way to express this
     * would be `if (a[0] + a[1] + a[2] >= 2)`.
     */
    if ((a[0] == 1 && a[1] == 1) || (a[1] == 1 && a[2] == 1) ||
        (a[0] == 1 && a[1] == 1 && a[2] == 1) || (a[0] == 1 && a[2] == 1)) {
      count1++;
    }
  }
  cout << endl << count1 << endl;
}
