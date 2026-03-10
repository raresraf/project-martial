/**
 * @file 282A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Bit++").
 * This program simulates a series of increment ("++") and decrement ("--")
 * operations on a single variable `x` and prints its final value.
 *
 * NOTE: This file contains a large number of pre-defined macros common in
 * competitive programming, which are not used in the core logic of this solution.
 *
 * @algorithm Simulation. The program iterates through each operation, using
 * `string::find` to check for the presence of "++" to determine whether to
 * increment or decrement a counter.
 *
 * @time_complexity O(N), where N is the number of operations. Each string operation
 * is effectively constant time.
 * @space_complexity O(1), using only a few variables.
 */

/* Monil Soni sonil06 */
/* DA-IICT  */

#include <bits/stdc++.h>

using namespace std;

typedef long long LL;
typedef vector<int> vi;
typedef vector<long long> vl;

// --- Boilerplate competitive programming macros ---
#define MOD 1000000007
#define _for(i, n) for (int i = 0; i < n; i++)
#define __for(i, n) for (int i = 1; i <= n; i++)
#define REP(i, s, n) for (int i = (s), _n = (n); i <= _n; i++)
#define FOR(i, n) for (int i = 0, _n = (n); i < _n; i++)
#define REPD(i, e, s) for (int i = (e), _s = (s); i >= _s; i--)
#define tr(container, it)                                                      \
  for (typeof(container.begin()) it = container.begin();                       \
       it != container.end(); it++)
#define pb push_back
#define mp make_pair
#define fi first
#define se second
#define ALL(x) x.begin(), x.end()

int main() {
  int n;
  scanf("%d", &n);
  string str;
  int x = 0; // The variable to be manipulated, initialized to 0.

  /**
   * @block
   * @brief Reads and executes `n` operations.
   * This loop iterates `n` times, processing one operation in each iteration.
   */
  for (int i = 0; i < n; i++) {
    cin >> str;
    /**
     * @brief Determines the operation type by searching for "++".
     * If "++" is found anywhere in the string, it's an increment.
     * Otherwise, it must be a decrement. `string::npos` is the value
     * returned by `find` when the substring is not found.
     */
    if (str.find("++") != string::npos)
      x++;
    else
      x--;
  }
  printf("%d\n", x);
  return 0;
}
