/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A (https://codeforces.com/problemset/problem/231/A).
 * @details This program counts the number of problems a team of three will solve.
 * The condition for solving a problem is that at least two of the three members
 * must be confident in the solution.
 *
 * @section
 *
 * Algorithm:
 * A direct simulation approach is used. The program iterates through all the problems.
 * For each problem, it reads the confidence levels of the three team members.
 * The sum of their confidence levels (represented as 0 or 1) is checked. If the sum
 * is 2 or more, it signifies a consensus, and a counter for solved problems is incremented.
 *
 * - Time Complexity: O(N), where N is the number of problems. The loop runs N times.
 * - Space Complexity: O(1), using a fixed amount of memory.
 */
#include <iostream>
using namespace std;
int main() {
  int num_prob, petya, vasya, tonya, result = 0;
  cin >> num_prob;
  /**
   * Block Logic: Loop through each of the 'num_prob' problems.
   * Invariant: 'result' contains the number of problems solved from the first 'i' problems.
   */
  for (int i = 0; i < num_prob; i++) {
    cin >> petya >> vasya >> tonya;
    /**
     * Block Logic: Check if at least two team members are confident.
     * This is done by summing up their boolean-like integer inputs (0 or 1).
     * Pre-condition: 'petya', 'vasya', and 'tonya' are either 0 or 1.
     */
    if (petya + vasya + tonya >= 2) {
      result += 1;
    }
  }
  cout << result << endl;
}
