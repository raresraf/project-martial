
// Problem : A. Shovels and Swords
// Contest : Codeforces - Educational Codeforces Round 89 (Rated for Div. 2)
// URL : https://codeforces.com/contest/1366/problem/A
// Memory Limit : 256.000000 MB
// Time Limit : 1000.000000 milisec
/**
 * @file 1366A.CPP
 * @brief Solves the "Shovels and Swords" problem (Codeforces 1366A).
 *
 * @details
 * The problem is to find the maximum number of tools (shovels or swords) that can be
 * crafted given a number of sticks and diamonds.
 * - Shovel: 2 sticks, 1 diamond
 * - Sword: 1 stick, 2 diamonds
 *
 * An optimal O(1) solution can be found with the formula `min(a, b, (a+b)/3)`, where
 * `a` and `b` are the counts of the two resources.
 *
 * This code, however, uses a non-obvious iterative greedy approach that repeatedly
 * crafts a batch of tools. It is heavily templated with competitive programming macros.
 *
 * Time Complexity: O(log(N+M)) per test case, where N and M are the resource counts,
 *                  as the number of resources decreases substantially in each iteration.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
// Collection of competitive programming macros for brevity.
#define ll long long int
#define pb push_back
#define F first
#define S second
#define mp make_pair
#define MOD 1000000007
#define vi vector<int>
#define vll vector<ll>
#define pll pair<ll, ll>
#define pii pair<int, int>
#define all(p) p.begin(), p.end()
#define mid(s, e) (s + (e - s) / 2)
#define eb emplace_back
#define ull unsigned long long
#define bug(x) cout << "  [ " #x << " = " << x << " ]" << endl;
#define KAMEHAMEHA ios_base::sync_with_stdio(0); // I/O optimization macro
#define RASENGAN                                                               \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);
using namespace std;
int main() {
  KAMEHAMEHA // Applying I/O optimization.
  int t;
  cin >> t;
  while (t--) {
    ll n, m; // n and m represent the counts of sticks and diamonds.
    cin >> n >> m;
    // Ensure n is always the larger of the two resource counts.
    if (n < m)
      swap(n, m);
    ll total = 0;

    /**
     * @brief Iterative greedy approach to craft tools.
     * The loop continues as long as it's possible to craft at least one of
     * the tool types (which require 2 of one resource and 1 of the other).
     */
    while (n >= 2 && m) {

      // This is a non-obvious heuristic to decide how many tools to craft in a batch.
      // It seems to try to make `temp` items of the type that costs (2n, 1m).
      ll temp = min((n / 4) + 1, m);
      n -= (temp * 2);
      m -= temp;
      if (temp == 0) // Break if no tools could be crafted in this iteration.
        break;
      total += temp;
      // After crafting, swap so that n remains the larger resource count for the next iteration.
      if (n < m)
        swap(n, m);
    }
    cout << total << endl;
  }
  return 0;
}
