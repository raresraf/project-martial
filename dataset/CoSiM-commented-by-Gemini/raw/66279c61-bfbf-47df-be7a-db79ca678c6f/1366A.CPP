/**
 * @file 1366A.CPP
 * @brief Solves the "Shovels and Swords" problem on Codeforces (Problem 1366A).
 * @details This program calculates the maximum number of tools (shovels or swords) that can be
 * crafted given a number of sticks (a) and diamonds (b).
 * - Shovel: 2 sticks, 1 diamond
 * - Sword: 1 stick, 2 diamonds
 *
 * Algorithm: Mathematical Analysis & Constraint-based Minimization.
 * Let x be the number of shovels and y be the number of swords. We want to maximize (x+y).
 * The constraints are:
 * 1. Stick constraint: 2*x + 1*y <= a
 * 2. Diamond constraint: 1*x + 2*y <= b
 *
 * Adding the two constraints gives: 3*x + 3*y <= a + b => 3*(x+y) <= a + b => x+y <= (a+b)/3.
 * Also, to make any tool, you need at least one stick and one diamond. So, the total number of
 * tools (x+y) cannot exceed the number of sticks 'a' or the number of diamonds 'b'.
 * This gives the constraints: x+y <= a and x+y <= b.
 *
 * Combining these, the maximum value of (x+y) is limited by the minimum of these three bounds.
 * Max tools = min(a, b, (a+b)/3).
 *
 * Time Complexity: O(1) for each test case, as it's a direct calculation.
 * Space Complexity: O(1).
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

//region --- Competitive Programming Boilerplate ---
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

// Macros for iterating through containers
#define TRvi(c, it) for (vi::iterator it = (c).begin(); it != (c).end(); it++)
#define TRvii(c, it) for (vii::iterator it = (c).begin(); it != (c).end(); it++)
#define TRmsi(c, it) for (msi::iterator it = (c).begin(); it != (c).end(); it++)

// Common constants
#define INF 2000000000 // 2 billion
// If you need to recall how to use memset:
#define MEMSET_INF 127 // about 2B
#define MEMSET_HALF_INF 63 // about 1B
// memset(dist, MEMSET_INF, sizeof dist); // useful to initialize shortest path
// distances memset(dp_memo, -1, sizeof dp_memo); // useful to initialize DP
// memoization table memset(arr, 0, sizeof arr); // useful to clear array of
// integers
//endregion

int main() {
  /**
   * Functional Utility: These lines decouple C++ streams from C standard streams
   * and untie cin from cout, significantly speeding up I/O operations, which is
   * critical in time-sensitive competitive programming environments.
   */
  ios::sync_with_stdio(false);
  cin.tie(0);
  
  int tc;
  cin >> tc; // Read the number of test cases.
  
  /**
   * @brief Block Logic: Processes each test case independently.
   */
  while (tc--) {
    int a, b; // a: number of sticks, b: number of diamonds
    cin >> a >> b;
    
    // The solution is the minimum of the three constraints derived from the problem:
    // 1. Total tools cannot exceed the number of sticks.
    // 2. Total tools cannot exceed the number of diamonds.
    // 3. Total tools cannot exceed the combined resources divided by three.
    int ans = min(a, min(b, (a + b) / 3));
    cout << ans << "\n";
  }
  return 0;
}
