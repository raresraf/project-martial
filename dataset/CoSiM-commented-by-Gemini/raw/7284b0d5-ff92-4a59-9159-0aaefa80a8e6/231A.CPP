/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A - "Team".
 *
 * This program determines how many problems a three-person team will solve.
 * A problem is solved if at least two members are confident in their solution.
 * The program reads the number of problems, then for each problem, it reads
 * the confidence level (1 for confident, 0 for not) of the three members.
 *
 * @algorithm Simple counting.
 * @time_complexity O(N), where N is the number of problems, as each problem
 *                  is processed in constant time within a single loop.
 * @space_complexity O(1).
 */
#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int count = 0;
  /**
   * @brief Processes each problem to determine if it will be solved.
   * This loop iterates through each of the 'n' problems.
   */
  for (int i = 0; i < n; i++) {
    int a, b, c;
    cin >> a >> b >> c;
    int c1 = 0;
    /**
     * @brief Counts the number of confident members for the current problem.
     */
    if (a == 1) {
      c1++;
    }
    if (b == 1) {
      c1++;
    }
    if (c == 1) {
      c1++;
    }
    /**
     * @brief Increments the total solution count if at least two members are confident.
     */
    if (c1 >= 2) {
      count++;
    }
  }
  cout << count << endl;
}