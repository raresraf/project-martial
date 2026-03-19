/**
 * @file raw/c5eb5311-e727-4c4f-a23d-4c3593d2bdb8/670A.CPP
 * @brief Calculates the sum of two minimum excluded (MEX) values for a given multiset.
 *
 * This program solves a problem over multiple test cases. For each test case,
 * it is given a multiset of integers and must calculate the sum of two values:
 * 1. The smallest non-negative integer that appears less than twice (MEX_2).
 * 2. The smallest non-negative integer that appears less than once (i.e., is absent, MEX_1).
 *
 * For example, for the multiset {0, 1, 2, 3}, MEX_2 is 0 and MEX_1 is 4. The sum is 4.
 * For the multiset {0, 0, 2, 3}, MEX_2 is 1 and MEX_1 is 1. The sum is 2.
 *
 * @algorithm
 * 1. For each test case, read the input integers into an array `A`.
 * 2. Count the frequency of each integer in `A` using a frequency array `cnt`.
 * 3. Iterate from `i = 0` upwards to find the two MEX-like values.
 * 4. A variable `rest` (initially 2) tracks how many values we still need to find.
 * 5. A variable `res` accumulates the sum.
 * 6. In each iteration `i`:
 *    - If `cnt[i] < rest`, it means `i` is one of the values we are looking for.
 *    - If `cnt[i] == 0`, we have found a number that is completely missing.
 *      `rest` values (either 1 or 2) are needed, and they are both `i` or greater.
 *      The missing sum is added (`rest * i`) and the function returns.
 *    - If `cnt[i] == 1`, we have found a number that appears only once. This
 *      can be our MEX_2. We add `i` to the result and decrement `rest` because
 *      one value has been found.
 * 7. If the loop completes, the remaining `rest` values are greater than any
 *    element in `A`, so we add `rest * (max_element + 1)` to the result.
 *
 * @time_complexity O(T * (N + M)), where T is the number of test cases, N is the
 *                  number of elements in the input array, and M is the maximum
 *                  value in the array.
 * @space_complexity O(M) for the frequency count array.
 */
#include <bits/stdc++.h>
using namespace std;

// --- Competitive Programming Macros ---
// These macros are shortcuts for common operations to speed up coding.
#define fi first
#define se second
#define mp make_pair
#define rep(i, n) for (int i = 0; i < (int)n; ++i)
#define repr(i, n) for (int i = (int)n; i > -1; --i)
#define REP(i, start, end, step)                                               
  for (int i = (int)start; i < (int)end; i += (int)step)
#define REPR(i, start, end, step)                                              
  for (int i = (int)start; i > (int)end; i += (int)step)
#define repc(i, c) for (auto i = c.begin(); i != c.end(); i++)
#define repcr(i, c) for (auto i = c.rbegin(); i != c.rend(); i++)
#define lowbit(x) ((x) & (-x)) // Gets the value of the least significant bit.

typedef long long ll;
typedef unsigned long long ull;

// Global variables to hold input for a single test case.
const int MAXN = 102;
int N, A[MAXN];

//-------------function-starts---------------------
//-------------function-ends-----------------------

void solve() {
  // Frequency array for numbers 0-101.
  int cnt[102] = {0};
  rep(i, N) cnt[A[i]]++;

  int rest = 2; // Number of MEX-like values we still need to find.
  int res = 0;  // The accumulating sum of the found values.
  int cur = 0;  // Tracks the next potential MEX value if the loop finishes.
  
  /**
   * @block
   * This loop iterates through non-negative integers starting from 0 to find
   * the two required MEX-like values.
   */
  rep(i, *max_element(A, A + N) + 1) {
    cur = i + 1;
    // Check if the count of number `i` is less than the number of values
    // we are still looking for.
    if (cnt[i] < rest) {
      if (cnt[i] == 0) {
        // `i` is the first number that is completely absent.
        // Both MEX_1 and MEX_2 must be at least `i`.
        // We add `i` for each of the `rest` values we still need.
        res += rest * i;
        cout << res << endl;
        return;
      } else { // cnt[i] == 1
        // `i` appears once. It can be our MEX_2 value.
        // We add `i` to the result and note that we only need to find one more value.
        res += i;
        rest--;
      }
    }
  }
  // If the loop completes, it means the missing values are larger than any
  // element in the array. `cur` holds the smallest integer not in the array.
  cout << res + rest * cur << endl;
}

int main() {
  int T;
  cin >> T;
  /**
   * @block
   * Main test case loop.
   */
  while (T--) {
    // Read input for the current test case.
    cin >> N;
    rep(i, N) cin >> A[i];
    solve();
  }
  return 0;
}
