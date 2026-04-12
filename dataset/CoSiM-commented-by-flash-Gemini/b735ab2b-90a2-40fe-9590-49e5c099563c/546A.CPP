/**
 * @file 546A.CPP
 * @brief Logic for calculating purchase deficits based on arithmetic price progression.
 * 
 * Logic: Calculates the total cost of buying 'w' items where the i-th item costs 
 * i * k. The total price is the sum of an arithmetic series: k * (1 + 2 + ... + w). 
 * The output is the required borrowing amount (Deficit), clamped at a minimum of 0.
 * 
 * Algorithm: Closed-form arithmetic series summation.
 * Formula: Total Cost = k * [w * (w + 1) / 2].
 * 
 * Time Complexity: $O(1)$ constant time calculation.
 * Space Complexity: $O(1)$ constant auxiliary space.
 */

#include <bits/stdc++.h>

using namespace std;
typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<ll> vll;

// Functional Utility: Macro-based loops and I/O for competitive programming.
#define forup(i, a, b) for (ll i = a; i < b; i++)
#define fordn(i, a, b) for (ll i = a; i >= b; i--)
#define fore(i, a, b) for (ll i = a; i <= b; i++)
#define rep(i, n) for (ll i = 0; i < n; i++)

#define dforup(i, a, b) for (i = a; i < b; i++)
#define dfordn(i, a, b) for (i = a; i >= b; i--)
#define dfore(i, a, b) for (i = a; i <= b; i++)
#define drep(i, a) for (i = 0; i < a; i++)

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
  ll k, n, w;
  
  /**
   * Functional Utility: Reads purchase parameters.
   * k: Price multiplier for the first item.
   * n: Available capital (initial budget).
   * w: Target quantity of items to purchase.
   */
  gl(k);
  gl(n);
  gl(w);

  /**
   * Logic: Sum of the first 'w' integers multiplied by k.
   * Uses the standard formula for triangular numbers.
   */
  ll totalcost = k * w * (w + 1) / 2;
  
  // Functional Utility: Calculates the net deficit.
  ll need = totalcost - n;
  
  /**
   * Logic: Non-negative clamping.
   * If current capital exceeds total price, no borrowing is required.
   */
  if (need < 0)
    need = 0;
  
  // Output the required financial supplement.
  pln(need);
  
  return 0;
}
