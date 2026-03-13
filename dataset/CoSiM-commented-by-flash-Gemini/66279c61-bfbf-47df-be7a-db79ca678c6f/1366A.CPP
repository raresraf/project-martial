/**
 * @file 1366A.CPP
 * @brief Solves the "Shovels and Swords" problem from Codeforces (1366A).
 * @details The program calculates the maximum number of tools (shovels or swords) that can be crafted given a certain number of sticks and diamonds.
 *
 * @section Crafting Rules
 * - Shovel: Requires 2 sticks and 1 diamond.
 * - Sword: Requires 1 stick and 2 diamonds.
 *
 * @section Algorithm
 * Let 'a' be the number of sticks and 'b' be the number of diamonds.
 * Let 'k' be the total number of tools to be crafted.
 *
 * The problem is to maximize 'k' subject to resource constraints.
 * 1. To make any tool, we need at least one stick and one diamond. Thus, the total number of tools cannot exceed the number of sticks or diamonds available.
 *    - `k <= a`
 *    - `k <= b`
 * 2. Let's analyze the total resources consumed. If we craft 'x' shovels and 'y' swords, where `x + y = k`:
 *    - Sticks used: `2x + y`
 *    - Diamonds used: `x + 2y`
 *    Summing the resource constraints: `(2x + y) + (x + 2y) <= a + b`, which simplifies to `3x + 3y <= a + b`, or `3k <= a + b`. This gives us the third constraint:
 *    - `k <= (a + b) / 3`
 *
 * To maximize 'k', we must satisfy all three constraints. The most restrictive constraint determines the maximum value of 'k'. Therefore, the answer is the minimum of the three upper bounds: `min(a, b, (a + b) / 3)`.
 *
 * @section Complexity
 * - Time Complexity: O(T) for T test cases, as each case is solved in O(1).
 * - Space Complexity: O(1).
 */
//============================================================================
// Name        : template.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <string.h> // for memset in CF judge.
#include <vector>
using namespace std;
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

int main() {
  /**
   * @brief Optimizes I/O operations for speed, a common practice in competitive programming.
   * `ios::sync_with_stdio(false)` decouples C++ streams from C's stdio.
   * `cin.tie(0)` unties cin from cout.
   */
  ios::sync_with_stdio(false);
  cin.tie(0);
  
  // Variable 'tc' holds the number of test cases to process.
  int tc;
  cin >> tc;

  /**
   * @brief Main loop to process each test case.
   */
  while (tc--) {
    // 'a' represents sticks, 'b' represents diamonds.
    int a, b;
    cin >> a >> b;

    /**
     * @brief Calculates the maximum number of tools based on the three binding constraints:
     * 1. Number of tools cannot exceed the number of sticks ('a').
     * 2. Number of tools cannot exceed the number of diamonds ('b').
     * 3. Number of tools cannot exceed one-third of the total resources ('(a + b) / 3').
     */
    int ans = min(a, min(b, (a + b) / 3));
    
    // Output the result for the current test case.
    cout << ans << "\n";
  }
  return 0;
}