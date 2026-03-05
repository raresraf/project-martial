/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A, "Team".
 * @details This program calculates the number of problems a three-person team
 * will solve, based on the number of members who are confident in the solution.
 *
 * Algorithm:
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter for solvable problems to zero.
 * 3. Loop `n` times to process each problem.
 * 4. In each iteration, read the three members' opinions (1 for confident, 0 for not).
 * 5. Sum the opinions. If the sum is 2 or greater, it means at least two
 *    members are confident, so the problem will be solved. Increment the counter.
 * 6. After the loop, print the final counter value.
 *
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */
#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;

  int cnt = 0;
  int a, b, c;
  /**
   * @brief This loop iterates through each of the `n` problems.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c;
    /**
     * @brief If the sum of opinions is 2 or more, the team will implement
     * the solution, so the counter is incremented.
     */
    if (a + b + c >= 2) {
      cnt++;
    }
  }

  cout << cnt << endl;
}
