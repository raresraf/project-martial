/**
 * @file 160A.CPP
 * @brief Solution for a checkout discount problem.
 * @author competitive-programmer
 *
 * @details This program implements a greedy algorithm to calculate the final cost
 * of a set of items after applying a special discount. The discount seems to be
 * of the "buy a bundle, get some items free" variety. The items are first
 * sorted by price, and the discount is applied to the most expensive items.
 *
 * Algorithm:
 * 1. Read an integer `m` and `m` subsequent numbers to determine a value `b`,
 *    which is the minimum of these numbers. This `b` can be interpreted as
 *    the size of a "bundle" for the discount.
 * 2. Read an integer `n` and `n` item prices into an array `a`.
 * 3. Calculate the initial total sum of all `n` items.
 * 4. Sort the item prices in non-decreasing order.
 * 5. Apply the discount greedily: Starting from the most expensive items,
 *    group them into bundles of size `b`. For each such bundle, the two
 *    most expensive items are made free (i.e., their price is subtracted
 *    from the total sum).
 * 6. Print the final calculated sum.
 *
 * Time Complexity: O(n log n) dominated by the sorting of item prices.
 * Space Complexity: O(n) to store the item prices.
 *
 * @note The problem appears to be mislabeled as 160A ("Twins"). The logic
 * implemented does not match that problem. It is a solution for a different,
 * discount-related problem.
 */
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <utility>
#include <vector>
//#include <ext/hash_map>

using namespace std;
// The following block contains common competitive programming macros and templates,
// many of which are not used in this specific solution.
using namespace __gnu_cxx;

#define REP(i, n) for (int i = 0; i < int(n); ++i)
#define REPV(i, n) for (int i = (n)-1; (int)i >= 0; --i)
#define FOR(i, a, b) for (int i = (int)(a); i < (int)(b); ++i)

#define FE(i, t)                                                               \
  for (__typeof((t).begin()) i = (t).begin(); i != (t).end(); ++i)
#define FEV(i, t)                                                              \
  for (__typeof((t).rbegin()) i = (t).rbegin(); i != (t).rend(); ++i)

#define two(x) (1LL << (x))
#define ALL(a) (a).begin(), (a).end()

#define pb push_back
#define ST first
#define ND second
#define MP(x, y) make_pair(x, y)

typedef long long ll;
typedef unsigned long long ull;
typedef unsigned int uint;
typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<string> vs;

template <class T> void checkmin(T &a, T b) {
  if (b < a)
    a = b;
}
template <class T> void checkmax(T &a, T b) {
  if (b > a)
    a = b;
}
template <class T> void out(T t[], int n) {
  REP(i, n) cout << t[i] << " ";
  cout << endl;
}
template <class T> void out(vector<T> t, int n = -1) {
  for (int i = 0; i < (n == -1 ? t.size() : n); ++i)
    cout << t[i] << " ";
  cout << endl;
}
inline int count_bit(int n) {
  return (n == 0) ? 0 : 1 + count_bit(n & (n - 1));
}
inline int low_bit(int n) { return (n ^ n - 1) & n; }
inline int ctz(int n) { return (n == 0 ? -1 : ctz(n >> 1) + 1); }
int toInt(string s) {
  int a;
  istringstream(s) >> a;
  return a;
}
string toStr(int a) {
  ostringstream os;
  os << a;
  return os.str();
}
// End of competitive programming template.

const int maxn = 111111;
int n, m;
int a[maxn];

int main() {
  // Read `m` and a list of `m` integers to determine `b`, the bundle size.
  cin >> m;
  int b = maxn;
  REP(i, m) {
    int x;
    scanf(" %d", &x);
    b = min(b, x);
  }
  
  // Read `n` and the `n` item prices.
  cin >> n;
  REP(i, n) scanf(" %d", a + i);
  
  // Sort the item prices in ascending order.
  sort(a, a + n);
  
  // Calculate the initial total cost.
  int res = 0;
  REP(i, n) res += a[i];

  /**
   * Block Logic: Apply the discount. The loop iterates from the most expensive
   * items downwards. For every `b` items, the two most expensive among them
   * are discounted.
   */
  for (int i = n - 1; i >= b;) {
    // Move the index `i` to skip `b` items. This defines a bundle.
    i -= b;
    /**
     * Inner loop discounts the two most expensive items in the current conceptual
     * bundle by subtracting their prices from the total result.
     */
    REP(j, 2) if (i >= 0) res -= a[i--];
  }
  
  // Print the final cost after discounts.
  cout << res << endl;

  return 0;
}
