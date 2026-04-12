/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of queue reordering based on local adjacency rules.
 * 
 * Logic: Simulates a queue where entities of type 'B' move behind entities of 
 * type 'G' if they are adjacent. Each time-step 't' allows each 'B' to migrate 
 * at most one position.
 * 
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: $O(T \times N)$ where T is the simulation duration and N is the queue length.
 * Space Complexity: $O(N)$ for the character sequence buffer.
 */

#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<ll, ll> pii;
typedef pair<double, ll> pdl;
typedef vector<ll> vll;

// Functional Utility: Standardized competitive programming macros for I/O and loops.
#define ff first
#define ss second
#define pb(x) push_back()
#define sl(x) scanf("%lld", &x)
#define sll(x, y) scanf("%lld%lld", &x, &y)
#define slll(x, y, z) scanf("%lld%lld%lld", &x, &y, &z)
#define pl(x) printf("%lld\n", x)
#define mp make_pair
ll mod = (ll)1e9 + 7;
const ll INF = (ll)1e18;
#define all(x) x.begin(), x.end()
#define s(x) ((ll)x.size())
#define tc                                                                     \
  ll t;                                                                        \
  cin >> t;                                                                    \
  while (t--)
#define FOR(i, j, n) for (ll i = j; i < n; i++)
#define ig cin.ignore()
#define fast_cin ios_base::sync_with_stdio(false)

/**
 * @brief Functional Utility: Modular exponentiation.
 * Note: Not utilized in the core queue logic but present in the source.
 */
ll power(ll a, ll b) {
  ll ret = 1;
  while (b) {
    if (b & 1)
      ret *= a;
    a *= a;
    if (ret >= mod)
      ret %= mod;
    if (a >= mod)
      a %= mod;
    b >>= 1;
  }
  return ret;
}

int main() {
  /**
   * Functional Utility: Optimizes standard I/O throughput.
   */
  fast_cin;

  ll n, t;
  // Reads total headcount (n) and temporal limit (t).
  if (!(cin >> n >> t)) return 0;
  
  string s;
  cin >> s;

  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * Invariant: Each while loop iteration represents one discrete time unit.
   */
  while (t--) {
    /**
     * Block Logic: Performs a single-pass sweep to resolve priority inversions.
     * Logic: Identifies 'B' preceding 'G' and performs a local state transition (swap).
     */
    for (ll i = 0; i < n - 1; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        /**
         * Inline: Skips the next index to prevent a single 'B' from migrating 
         * through multiple 'G's in a single time step (atomic movement).
         */
        i++;
      }
    }
  }
  
  // Output the final state of the queue sequence.
  cout << s << "\n";
  
  return 0;
}
