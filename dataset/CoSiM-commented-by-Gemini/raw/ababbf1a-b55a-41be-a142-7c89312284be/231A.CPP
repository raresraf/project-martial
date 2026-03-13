/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 * @details The program counts the number of problems a three-person team will
 * decide to solve. The team solves a problem if at least two of the three
 * members are confident in the solution.
 *
 * Algorithm:
 * 1. Read the total number of problems, 'n'.
 * 2. Initialize a counter for solvable problems to zero.
 * 3. Loop 'n' times, once for each problem:
 *    a. Read the three members' confidence levels (1 for confident, 0 for not).
 *    b. Calculate the sum of the three confidence levels.
 *    c. If the sum is 2 or greater, it means at least two members are confident,
 *       so increment the counter.
 * 4. After the loop, print the total count of solvable problems.
 *
 * Time Complexity: O(N), where N is the number of problems, due to the single loop.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * Functional Utility: Decouples C++ streams from standard C streams and
   * unties cin from cout to optimize I/O performance, which is a common
   * practice in competitive programming.
   */
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  int n, count = 0, a1, a2, a3;
  cin >> n;
  for (int i = 0; i < n; ++i) {
    cin >> a1 >> a2 >> a3;
    if (a1 + a2 + a3 > 1)
      count++;
  }
  cout << count << "\n";
}