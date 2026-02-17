/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A, team consensus counting.
 * @details This program counts how many problems a team will solve, where a
 * problem is solved if at least two of the three team members are confident.
 * This version uses nested loops and an array to read member confidence.
 *
 * @section
 *
 * Algorithm:
 * The program is structured to handle multiple test sets. For each set, it reads
 * the number of problems, `n`. It then loops `n` times. In each of these
 * iterations, it uses another inner loop to read the three confidence values (0 or 1)
 * into an array. A counter tallies how many of these three values are '1'. If this
 * counter is greater than one, a final "answer" counter is incremented.
 *
 * - Time Complexity: O(S * N * 3), where S is the number of test sets and N is the
 *   number of problems per set. Since S=1 and N<=1000 for the original problem,
 *   this is effectively O(N).
 * - Space Complexity: O(1).
 *
 * Note: The file includes a large number of pre-processor macros, a common
 * practice in competitive programming.
 */

/***

            Bismillahir Rahmanir Rahim
            Read the name of Allah, who created you!!!
            Author : Shah Newaj Rabbi Shishir
            Department of CSE, City University, Bangladesh.

***/

#include <bits/stdc++.h>
using namespace std;

#define sf scanf
#define pf printf
#define scase sf("%d", &tc)
#define sn sf("%d", &n)
#define whilecase while (tc--)
#define eof while (cin >> n)
#define forloop for (pos = 1; pos <= tc; pos++)
#define arrayloop (i = 0; i < n; i++)
#define cinstr cin >> str
#define getstr getline(cin, str)
#define pcase pf("Case %d: ", pos)
#define pb push_back
#define in insert
#define llu unsigned long long
#define lld long long
#define u unsigned int

int main(void) {
  int n, arr[3], i, ans, c;

  /**
   * Block Logic: Handles multiple sets of test cases, though the problem
   * typically only has one. 'n' is the number of problems in a set.
   */
  while (cin >> n) {
    ans = 0; // Counter for solved problems in the current set.

    /**
     * Block Logic: Iterates through each of the 'n' problems.
     */
    while (n--) {
      c = 0; // Counter for confident members for the current problem.

      /**
       * Block Logic: Reads the three confidence values into an array and
       * counts how many members are confident (value is 1).
       */
      for (i = 0; i < 3; i++) {
        cin >> arr[i];

        if (arr[i] == 1)
          c++;
      }

      /**
       * Block Logic: Checks if the consensus is met (at least two members).
       */
      if (c > 1)
        ans++;
    }

    cout << ans << endl;
  }

  return 0;
}
