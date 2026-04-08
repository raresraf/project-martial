/**
 * @file 670A.CPP
 * @brief This program solves a problem related to finding the sum of the two smallest 
 * non-negative integers that have a limited occurrence in an input array. 
 * This is a concept often related to the "minimum excluded value" (MEX) in competitive programming.
 *
 * @algorithm
 * For each test case:
 * 1. Read an array `A` of `N` integers.
 * 2. Count the frequency of each number in `A` and store it in `cnt`.
 * 3. Find the first smallest non-negative integer `m1` that appears less than twice in `A`.
 * 4. Find the second smallest non-negative integer `m2` that appears less than once in `A` (after considering `m1`).
 * 5. The result is the sum of these two values. The logic iterates through integers `i` from 0 upwards.
 *    - It uses a `rest` counter, initialized to 2, to track how many "missing" numbers it needs to find.
 *    - If `cnt[i]` is 0, it means `i` is missing. The contribution to the sum depends on the current `rest` value.
 *    - If `cnt[i]` is 1, it "uses up" one of the required missing numbers.
 *
 * @time_complexity O(T * (N + M)), where T is the number of test cases, N is the number of elements, 
 * and M is the maximum value in the input array.
 * @space_complexity O(M) for the frequency count array.
 */
#include <bits/stdc++.h>
using namespace std;
#define fi first
#define se second
#define mp make_pair
#define rep(i, n) for (int i = 0; i < (int)n; ++i)
#define repr(i, n) for (int i = (int)n; i > -1; --i)
#define REP(i, start, end, step)                                               \
  for (int i = (int)start; i < (int)end; i += (int)step)
#define REPR(i, start, end, step)                                              \
  for (int i = (int)start; i > (int)end; i += (int)step)
#define repc(i, c) for (auto i = c.begin(); i != c.end(); i++)
#define repcr(i, c) for (auto i = c.rbegin(); i != c.rend(); i++)
#define lowbit(x) ((x) & (-x))
typedef long long ll;
typedef unsigned long long ull;
// global variables
const int MAXN = 102;
int N; // Number of elements in the input array
int A[MAXN]; // The input array

//-------------function-starts---------------------
//-------------function-ends-----------------------

/**
 * @brief Solves a single test case for the MEX-like problem.
 */
void solve() {
  int cnt[102] = {0}; // Frequency map for numbers in the input array.
  // Populate the frequency map.
  rep(i, N) cnt[A[i]]++;
  
  int rest = 2; // Represents the number of smallest non-present integers we are looking for.
  int res = 0; // Stores the sum of the found integers.
  int cur = 0; // Tracks the current integer being checked.

  /**
   * @block_logic Iteratively find the two smallest non-negative integers that are "missing" from the multiset.
   * @pre `cnt` array is populated with frequencies.
   * @invariant `res` holds the sum of the "missing" integers found so far. `rest` holds the count of integers still to find.
   * @post The loop terminates once two "missing" numbers are found and their sum is calculated.
   */
  rep(i, *max_element(A, A + N) + 2) { // Iterate up to max value + 2 to handle edge cases
    cur = i;
    if(cnt[i] == 0) { // If number `i` is completely absent
        res += rest * i; // Add `i` to the result `rest` times.
        cout << res << endl;
        return;
    } else if (cnt[i] < rest) { // If `i` is present but not enough to fulfill the 'rest' count
        res += i; // Add `i` once.
        rest--; // We have found one of the numbers we are looking for.
    }
  }
  // This part is reached if the loop completes, which implies the missing numbers are >= max value in A.
  cout << res + rest * cur << endl;
}

int main() {
  int T; // Number of test cases
  cin >> T;
  while (T--) {
    // Read input for each test case
    cin >> N;
    rep(i, N) cin >> A[i];
    solve();
  }
  return 0;
}