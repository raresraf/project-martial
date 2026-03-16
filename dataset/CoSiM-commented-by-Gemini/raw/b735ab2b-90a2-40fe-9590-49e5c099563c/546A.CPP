/**
 * @file 546A.CPP
 * @brief Solution to the "Soldier and Bananas" problem (Codeforces 546A).
 *
 * @details
 * This program calculates the amount of money a soldier needs to borrow to buy `w`
 * bananas. The cost increases linearly: the first banana costs `k`, the second `2k`,
 * and so on. The code uses a direct mathematical formula to find the total cost
 * and then determines the deficit.
 *
 * The total cost is the sum of an arithmetic series:
 *   Total Cost = k * (1 + 2 + ... + w)
 * The sum `1 + 2 + ... + w` is calculated using the formula `w * (w + 1) / 2`.
 * Thus, Total Cost = `k * w * (w + 1) / 2`.
 *
 * The amount to borrow (`need`) is the `Total Cost - n`, where `n` is the initial
 * amount of money. If the soldier has enough money, the amount needed is 0.
 *
 * This implementation is characterized by heavy use of C++ macros for I/O and loops,
 * a common idiom in competitive programming to reduce boilerplate code.
 *
 * Algorithm: Direct Mathematical Calculation
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
#include <bits/stdc++.h>

using namespace std;
typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<ll> vll;

// Macros for common loop structures
#define forup(i, a, b) for (ll i = a; i < b; i++)
#define fordn(i, a, b) for (ll i = a; i >= b; i--)
#define fore(i, a, b) for (ll i = a; i <= b; i++)
#define rep(i, n) for (ll i = 0; i < n; i++)

#define dforup(i, a, b) for (i = a; i < b; i++)
#define dfordn(i, a, b) for (i = a; i >= b; i--)
#define dfore(i, a, b) for (i = a; i <= b; i++)
#define drep(i, a) for (i = 0; i < a; i++)

// Macros for vector operations and I/O
#define all(v) v.begin(), v.end()
#define gi(n) scanf("%d", &n)
#define gl(n) cin >> n
#define pin(x) printf("%d\n", x)
#define pln(x) cout << x
#define pn printf("\n")
#define ps printf(" ")

#define fs first
#define sc second
#define pb push_back

#define repIt(v) for (auto it = v.begin(); it != v.end(); ++it) // C++11
#define INPFILE freopen("input.in", "r", stdin)

int main() {
  // k: cost of the first banana
  // n: initial money the soldier has
  // w: number of bananas to buy
  ll k, n, w;
  gl(k);
  gl(n);
  gl(w);

  // Calculate the total cost using the formula for the sum of an arithmetic series.
  ll totalcost = k * w * (w + 1) / 2;

  /**
   * @brief Calculate the deficit.
   * @details The amount needed is the difference between the total cost and the money
   * the soldier already has. If this value is negative (i.e., the soldier has
   * a surplus), the amount needed is 0.
   */
  ll need = totalcost - n;
  if (need < 0)
    need = 0;

  pln(need); // Print the result.
}