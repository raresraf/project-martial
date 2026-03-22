/**
 * @file 50A.CPP
 * @brief Calculates the maximum number of dominoes on a rectangular board.
 *
 * This file contains a solution for the "Domino Piling" problem. The goal is to
 * find the maximum number of 2x1 dominoes that can fit on an M x N board.
 * The file includes a large competitive programming template with many unused
 * macros and typedefs.
 *
 * @b Algorithm: Complex Calculation for a Simple Problem
 *   The code uses a complicated and redundant method to arrive at the answer.
 *   It calculates the result in two different ways (`first` and `second`), which
 *   both mathematically simplify to `floor((M * N) / 2)`, and then takes the
 *   maximum. The direct solution is simply `(m * n) / 2` using integer division.
 *
 * @b Time_Complexity: O(1).
 * @b Space_Complexity: O(1).
 */
//============================================================================
// Name        : template.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <bits/stdc++.h>
using namespace std;
// --- Start of Competitive Programming Template ---
#define _CRT_SECURE_NO_DEPRECATE // suppress some compilation warning messages
                                 // (for VC++ users)
// Shortcuts for "common" data types in contests
typedef long long ll;
typedef vector<int> vi;
typedef pair<int, int> ii;
typedef vector<ii> vii;
typedef set<int> si;
typedef map<string, int> msi;
// To simplify repetitions/loops, Note: define your loop style and stick with
// it!
#define x first
#define y second
#define pb push_back
#define mp make_pair
#define REP(i, a, b)                                                           \
  for (int i = int(a); i <= int(b); i++) // a to b, and variable i is local!
#define forn(i, n) for (int i = 0; i < (n); i++)
#define TRvi(c, it) for (vi::iterator it = (c).begin(); it != (c).end(); it++)
#define TRvii(c, it) for (vii::iterator it = (c).begin(); it != (c).end(); it++)
#define TRmsi(c, it) for (msi::iterator it = (c).begin(); it != (c).end(); it++)
#define INF 2000000000 // 2 billion
// If you need to recall how to use memset:
#define MEMSET_INF 127 // about 2B
#define MEMSET_HALF_INF 63 // about 1B
// memset(dist, MEMSET_INF, sizeof dist); // useful to initialize shortest path
// distances memset(dp_memo, -1, sizeof dp_memo); // useful to initialize DP
// memoization table memset(arr, 0, sizeof arr); // useful to clear array of
// integers
// --- End of Competitive Programming Template ---

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  int m, n;
  cin >> m >> n;

  int a = max(m, n);
  int b = min(m, n);
  int ans = 0;
  if (a < 2 || b < 1)
    ans = 0;
  else {
    /**
     * @brief The following is a convoluted way to calculate floor((m*n)/2).
     * It considers tiling the board along its shorter and longer axes separately.
     */
     
    // First calculation method based on the shorter side 'b'.
    int first = 0;
    if ((b & 1)) { // If 'b' is odd
      first = (a * (b / 2)) + ((b % 2) * (a / 2));
    } else { // If 'b' is even
      first = (a * (b / 2));
    }

    // Second, redundant calculation method based on the longer side 'a'.
    int second = 0;
    if ((a & 1)) { // If 'a' is odd
      second = (b * (a / 2)) + ((a % 2) * (b / 2));
    } else { // If 'a' is even
      second = (b * (a / 2));
    }
    
    // Both `first` and `second` yield the same result, floor((m*n)/2).
    ans = max(second, first);
  }

  // A much simpler solution is `ans = (m * n) / 2;`
  cout << ans << "\n";
  return 0;
}
