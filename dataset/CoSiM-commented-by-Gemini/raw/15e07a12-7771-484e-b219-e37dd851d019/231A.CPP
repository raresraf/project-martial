/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem, using a competitive programming template.
 * @details This program counts the number of problems a team of three will solve, based on
 * the rule that a problem is solved if at least two members are confident. The code is
 * embedded within a large template of macros and helper functions common in competitive
 * programming, but the core logic is a straightforward, space-efficient implementation.
 *
 * @section algorithm Algorithm
 * 1. Read the number of problems, `n`, using a custom macro `gi(n)` (equivalent to `scanf("%d", &n)`).
 * 2. Initialize an answer counter `a` to 0.
 * 3. Loop `n` times using a `rep(i, n)` macro (equivalent to `for (int i = 0; i < n; ++i)`).
 *    a. For each problem, initialize a temporary counter `c` to 0 to sum the votes.
 *    b. Loop 3 times for the three team members using another `rep` macro.
 *    c. In each inner loop, read a member's decision (0 or 1) and add it to `c`.
 *    d. After reading the three decisions, if `c` is 2 or more, increment the answer counter `a`.
 * 4. Print the final answer `a` using a `pin(a)` macro (equivalent to `printf("%d\n", a)`).
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N), for iterating through the `N` problems.
 * - Space Complexity: O(1), as problems are processed one by one without storing all inputs.
 */
#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string.h>
#include <string>
#include <utility>
#include <vector>

// NOTE: The following is a standard template for competitive programming.
// It includes custom macros for loops and I/O, typedefs, and helper functions.
using namespace std;
typedef long long LL;
typedef pair<int, int> pii;

#define forup(i, a, b) for (int i = a; i < b; ++i)
#define fordn(i, a, b) for (int i = a; i > b; --i)
#define rep(i, a) for (int i = 0; i < a; ++i)

#define dforup(i, a, b) for (i = a; i < b; ++i)
#define dfordn(i, a, b) for (i = a; i > b; --i)
#define drep(i, a) for (i = 0; i < a; ++i)

#define slenn(s, n)                                                            \
  for (n = 0; s[n] != 13 and s[n] != 0; ++n)                                   \
    ;                                                                          \
  s[n] = 0

#define gi(x) scanf("%d", &x)
#define gl(x) scanf("%lld", &x)
#define gd(x) scanf("%lf", &x)
#define gs(x) scanf("%s", x)

#define pis(x) printf("%d ", x)
#define pin(x) printf("%d\n", x)
#define pls(x) printf("%lld ", x)
#define pln(x) printf("%lld\n", x)
#define pds(x) printf("%.12f ", x)
#define pdn(x) printf("%.12f\n", x)
#define pnl() printf("\n")

#define fs first
#define sc second
#define ll long long
#define pb push_back
#define MOD 1000000007
#define limit 10000005
#define INF 1000000000
#define ull unsigned long long
using namespace std; // Redundant `using namespace std;`
ull mod_pow(ull num, ull pow, ull mod) {
  ull test, n = num;
  for (test = 1; pow; pow >>= 1) {
    if (pow & 1)
      test = ((test % mod) * (n % mod)) % mod;
    n = ((n % mod) * (n % mod)) % mod;
  }
  return test; /* note this is potentially lossy */
}
// while((getchar())!='\n'); //buffer clear
ll gcd(ll a, ll b) {
  ll r;
  while (b) {
    r = a % b;
    a = b;
    b = r;
  }
  return a;
}

int main() {
  int n, a = 0;
  // gi(n) is a macro for scanf("%d", &n);
  gi(n);
  // rep(i, n) is a macro for for(int i=0; i<n; ++i)
  rep(i, n) {
    int c = 0, x;
    // Inner loop to read the 3 decisions for the current problem.
    rep(j, 3) {
      gi(x);
      c += x;
    }
    // If two or more members are confident, increment the answer count.
    if (c >= 2)
      a++;
  }
  // pin(a) is a macro for printf("%d\n", a);
  pin(a);
}
