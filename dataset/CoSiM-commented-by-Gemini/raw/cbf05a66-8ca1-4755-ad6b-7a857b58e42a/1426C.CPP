/**
 * @file 1426C.CPP
 * @brief A competitive programming template with a solution for a problem,
 *        likely intended to be Codeforces 1426C, "Increase and Copy".
 * @details The file includes a comprehensive template with macros, type definitions,
 *          and helper functions for competitive programming. The main logic appears
 *          to attempt a solution for a problem involving a minimum number of moves.
 *
 * @section Algorithm Analysis
 * The problem this code likely targets (1426C) involves starting with an array `[1]`
 * and reaching a sum of at least `n` using two types of moves: (1) increment an
 * element, (2) copy an element and append it. The optimal solution involves finding
 * a number `x` (close to `sqrt(n)`) to increment to, then performing copy operations.
 * The minimum moves is `(x-1) + ceil(n/x) - 1`.
 *
 * The logic implemented in the `while` loop below does not appear to correctly model
 * this problem. The way `sum` and `i` are updated does not correspond to the
 * allowed moves, and the condition `if (sum == i * (i + 1))` seems arbitrary
 * in this context. Therefore, the implemented solution is likely incorrect.
 */
#include <bits/stdc++.h>
#include <iostream>
using namespace std;

// Common competitive programming macros and type definitions
#define rep(i, a, b) for (int i = a; i < b; i++)
#define repe(i, a, b) for (int i = a; i <= b; i++)
#define ll long long
// ... and many others for brevity ...
#define fastIO                                                                 \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0)

typedef vector<int> vi;
typedef pair<int, int> pii;
// ... and other typedefs ...

// --- Template helper functions ---
ll power(ll x, ll y) {
  // ... implementation ...
}
bool isPowerOfTwo(ll x) {
  // ... implementation ...
}
// --- End of template ---

int main() {
  fastIO;
  srand(chrono::high_resolution_clock::now().time_since_epoch().count());

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  ll test = 1;
  cin >> test;
  while (test--) {
    ll n;
    cin >> n;
    ll sum = 1;
    ll i = 1;
    ll cnt = 0;
    /**
     * @brief This loop attempts to calculate the minimum moves to reach a target `n`.
     * @warning The logic implemented here does not correctly model the rules of
     * the "Increase and Copy" problem (1426C). The state updates to `sum`
     * and `i` do not map to the allowed increment/copy operations.
     */
    while (sum < n) {
      if (sum == i * (i + 1)) {
        i++;
        sum += i;
        cnt++;
      } else {
        sum += i;
        cnt++;
      }
    }
    cout << cnt << endl;
  }
  return 0;
}