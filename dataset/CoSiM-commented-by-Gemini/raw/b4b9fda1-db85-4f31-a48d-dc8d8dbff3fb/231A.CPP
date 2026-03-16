/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem on Codeforces.
 * @url https://codeforces.com/problemset/problem/231/A
 *
 * @details
 * The problem is about a programming contest where teams of three decide to
 * implement a problem only if at least two of them are sure about the solution.
 * The program reads the number of problems and for each problem, the three
 * members' opinions (1 for sure, 0 for not sure). It then counts how many
 * problems the team will implement.
 *
 * @algorithm
 * The program iterates through each problem, reads the three opinions, and sums
 * them up. If the sum is 2 or greater, it means at least two members are sure,
 * so a counter for implemented problems is incremented.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of problems, as the program
 * iterates through each problem once.
 * Space Complexity: O(1), as it only uses a few variables to store the counts
 * and opinions.
 */
#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;
  int a, b, c;
  int result = 0;

  /**
   * @brief Iterate through each problem to check the team's confidence.
   * For each of the `n` problems, read the opinions of the three members.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c;

    /**
     * @brief Check if at least two members are sure.
     * If the sum of their opinions (which are either 0 or 1) is 2 or more,
     * the team will implement the problem.
     */
    if ((a + b + c) >= 2) {
      result++;
    }
  }

  cout << result << endl;

  return 0;
}
