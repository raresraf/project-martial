
/*
   By Anmol Gupta :)
*/
#include <bits/stdc++.h>
using namespace std;

/**
 * @file 546A.CPP
 * @brief Solution to Codeforces problem 546A, "Soldier and Bananas".
 * @details The program calculates the amount of money a soldier needs to borrow.
 *
 * Variable Mapping Note:
 * The variable names in this code do not match the problem statement.
 * - `n` in code corresponds to `k` (base cost) in the problem.
 * - `k` in code corresponds to `n` (initial money) in the problem.
 * - `d` in code corresponds to `w` (number of bananas) in the problem.
 *
 * Algorithm:
 * The total cost is the sum `k + 2k + ... + wk`, which is `k * (1 + ... + w)`.
 * The code calculates this using the arithmetic series formula: `k * w * (w+1) / 2`.
 * It then compares this total cost to the soldier's initial money to find the
 * amount that needs to be borrowed.
 *
 * Time Complexity: O(1), due to the direct mathematical formula.
 * Space Complexity: O(1).
 */

#pragma GCC optimize("O3")
// --- Boilerplate and macros for competitive programming ---
#define pb push_back
#define debug(x) cout << #x << " " << x << endl;
#define mp make_pair
#define ff first
#define ss second
#define allunique(v) v.erase(unique(v.begin(), v.end()), v.end());
#define endl "\n"
#define MOD 1000000007
#define MOD9 1000000009
#define gc getchar_unlocked
#define ms(s, n) memset(s, n, sizeof(s))
#define prec(n) fixed << setprecision(n)
#define eps 0.000001
#define pll pair<ll, ll>
#define all(v) v.begin(), v.end()
#define bolt ios::sync_with_stdio(0)
#define forr(i, n) for (ll i = 0; i < n; i++)
#define eof (scanf("%d", &n)) != EOF
typedef long long ll;
// --- End of boilerplate ---

int main() {
  // Fast I/O setup.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  // Variable names are mapped from the problem statement:
  // n: base cost `k`
  // k: initial money `n`
  // d: number of bananas `w`
  ll n, k, d, tot;
  cin >> n >> k >> d;

  /**
   * Block Logic: Calculate total cost using the arithmetic series formula.
   * `tot = n * (d * (d+1) / 2)` is `k * (w * (w+1) / 2)`.
   */
  tot = n * d * (d + 1);
  tot = tot / 2;

  /**
   * Block Logic: Determine the amount to borrow.
   * Compares total cost (`tot`) with available money (`k` in code).
   */
  if (tot > k)
    cout << tot - k;
  else
    cout << "0";
}
