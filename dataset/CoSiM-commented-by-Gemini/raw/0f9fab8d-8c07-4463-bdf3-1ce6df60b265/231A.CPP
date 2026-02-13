/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem (Codeforces 231A).
 * @author Yadi Qian
 *
 * @details This program calculates the number of problems a three-person team will
 * decide to implement. The team decides to solve a problem if at least two of
 * its members are sure about the solution.
 *
 * Algorithm:
 * 1. Read the total number of problems, `num`.
 * 2. Initialize a counter, `implement`, to zero. This will store the number of
 *    problems the team will solve.
 * 3. Loop `num` times, once for each problem.
 * 4. In each iteration, read the three opinions (0 or 1) of the team members.
 * 5. Count how many members are sure (i.e., how many opinions are 1).
 * 6. If the count of sure members is two or more, increment the `implement` counter.
 * 7. After the loop, print the final value of `implement`.
 *
 * Time Complexity: O(num), where `num` is the number of problems.
 * Space Complexity: O(1).
 */

#include <bits/stdc++.h>

using namespace std;

// Common competitive programming type definitions.
const double pi = acos(-1.0);
typedef pair<int, int> PII;
typedef vector<int> VI;
typedef vector<string> VS;
typedef vector<PII> VII;
typedef vector<VI> VVI;
typedef map<int, int> MPII;
typedef set<int> SETI;
typedef multiset<int> MSETI;
typedef long int int32;
typedef unsigned long int uint32;
typedef long long int int64;
typedef unsigned long long int uint64;

int main() {
  // Optimizes C++ standard streams for faster I/O in competitive programming.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  // `num` is the total number of problems to be evaluated.
  int num;
  cin >> num;
  // `implement` will count the number of problems the team decides to solve.
  int implement = 0;

  /**
   * Block Logic: This loop iterates through each of the `num` problems.
   */
  for (int i = 0; i < num; i++) {
    // a, b, c represent the opinions of the three team members (1 for sure, 0 for not).
    int a, b, c;
    cin >> a >> b >> c;
    // `sure` counts how many members are sure about the current problem.
    int sure = 0;

    // Tally the number of 'sure' opinions.
    if (a == 1)
      sure++;
    if (b == 1)
      sure++;
    if (c == 1)
      sure++;

    /**
     * Decision Logic: The problem is implemented if at least two members are sure.
     */
    if (sure >= 2) {
      implement++;
    }
  }

  // Output the total number of problems that will be implemented.
  cout << implement;

  return 0;
}
