/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem (Codeforces 231A).
 * @details This program calculates the number of problems a three-person team will
 *          implement. The team decides to solve a problem if at least two members
 *          are sure about the solution.
 *
 * @algorithm The algorithm iterates through each problem, summing up the confidence
 *            levels (represented as 1 for sure, 0 for not sure) of the three team
 *            members. If the sum is 2 or greater, it means at least two members are
 *            confident, and a counter for implemented problems is incremented.
 *
 * @time_complexity O(N), where N is the number of problems, as we process each
 *                  problem once.
 * @space_complexity O(1), as we only need to store a few variables for the calculation.
 */
#include <iostream>

using namespace std;

int main() {
  int test, count = 0;
  cin >> test;

  /**
   * @brief This loop iterates through each of the 'test' problems.
   */
  while (test--) {
    int x, y, z;
    cin >> x >> y >> z;

    auto sum = x + y + z;
    /**
     * @brief Check if at least two members are sure about the solution.
     * The sum of their confidence levels (0 or 1) will be >= 2 if that is the case.
     */
    if (sum >= 2)
      count++;
  }

  // Output the total number of problems the team will implement.
  cout << count << endl;

  return 0;
}
