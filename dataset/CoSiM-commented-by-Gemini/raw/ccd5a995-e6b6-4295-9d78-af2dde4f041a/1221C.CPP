/**
 * @file 1221C.CPP
 * @brief Solution for "Perfect Team" (Codeforces 1221C).
 *
 * @details
 * The problem is to find the maximum number of three-person teams that can be formed
 * from `c` programmers, `m` mathematicians, and `x` individuals with no specialization.
 * A team must consist of one programmer and one mathematician. The third person can be
 * anyone (a programmer, a mathematician, or a no-specialization individual).
 *
 * @section algorithm Algorithm
 * Let `k` be the number of teams. To form `k` teams, we need:
 * 1. `k` programmers, so `k <= c`.
 * 2. `k` mathematicians, so `k <= m`.
 * 3. The total number of people involved is `3k`. The total available is `c + m + x`.
 *    So, `3k <= c + m + x`, which means `k <= (c + m + x) / 3`.
 *
 * Combining these, the maximum number of teams is `k = min(c, m, (c + m + x) / 3)`.
 * The code implements this logic via a case analysis.
 *
 * @section complexity Complexity
 * - Time Complexity: O(1) for each test case.
 * - Space Complexity: O(1).
 */
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

// --- Boilerplate from a competitive programming template ---
using namespace std;
using namespace __gnu_pbds;
typedef long long ll;
typedef long double ld;
typedef complex<ld> cd;
typedef pair<int, int> pi;
typedef pair<ll, ll> pl;
typedef pair<ld, ld> pd;
typedef vector<int> vi;
typedef vector<ld> vd;
typedef vector<ll> vl;
typedef vector<pi> vpi;
typedef vector<pl> vpl;
typedef vector<cd> vcd;
typedef vector<vector<int>> vvi;
typedef vector<vector<ll>> vvl;
template <class T>
using Tree =
    tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
#define FOR(i, a, b) for (int i = a; i < (b); i++)
#define F0R(i, a) for (int i = 0; i < (a); i++)
#define FORd(i, a, b) for (int i = (b)-1; i >= a; i--)
#define F0Rd(i, a) for (int i = (a)-1; i >= 0; i--)
#define sz(x) (int)(x).size()
#define mp make_pair
#define pb push_back
#define f first
#define s second
#define lb lower_bound
#define ub upper_bound
#define all(x) x.begin(), x.end()
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
const int MOD = 1000000007;
const ll INF = 1e18;
const int MX = 100001;
// --- End of Boilerplate ---

int main() {
  int q; // Number of test cases.
  cin >> q;
  while (q--) {
    int c, m, x; // c: programmers, m: mathematicians, x: others
    cin >> c >> m >> x;

    /**
     * @brief The solution is equivalent to finding `min(c, m, (c + m + x) / 3)`.
     * The code implements this through a direct, if complex, formula.
     */

    // Case 1: The number of 'others' (x) is not a primary bottleneck.
    // If x is large, the answer is limited by min(c, m). This case is a
    // specific instance covered by the general formula.
    if (x >= m || x >= c) {
      cout << min(m, c) << endl;
      continue;
    }

    // Case 2: The general case, where x is a potential bottleneck.
    // We can form `x` teams using all 'other' individuals.
    // We are left with `c-x` programmers and `m-x` mathematicians.
    // The number of additional teams is limited by the remaining members.
    int rc = c - x;
    int rm = m - x;
    if (rc < rm)
      swap(rc, rm); // Ensure rc is the larger count.
    
    // This expression calculates the number of additional teams that can be formed
    // from the remaining `rc` and `rm` individuals. It is a non-obvious
    // formula equivalent to min(rm, (rc + rm) / 3).
    cout << x + min(rm, rc / 2 + max(0, (rm - rc / 2 + (rc & 1)) / 3)) << endl;
  }
}
