/**
 * @file 266B.CPP
 * @brief Solution to the "Queue at the School" problem (Codeforces 266B).
 *
 * @details
 * This program simulates the state of a school queue of boys ('B') and girls ('G')
 * over `t` time steps. In each step, any boy standing directly in front of a girl
 * swaps places with her.
 *
 * This implementation is characterized by a heavy use of macros and typedefs, a common
 * style in competitive programming to accelerate coding. The core logic is a nested
 * loop structure. The outer loop simulates the time steps, and the inner loop
 * performs a single pass over the queue to execute all swaps for that step.
 *
 * After a 'BG' pair is swapped at index `i`, the loop counter `i` is incremented.
 * The loop then continues from the next index `i+1`. This correctly models the
 * "simultaneous" nature of the swaps within a single time step, as no child moves
 * more than one position per pass.
 *
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: O(t * n), where `t` is the number of time steps and `n` is the
 *                  queue length.
 * Space Complexity: O(n), to store the queue string.
 */
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<ll, ll> pii;
typedef pair<double, ll> pdl;
typedef vector<ll> vll;
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
  ll t;
  cin >> t;
  while (t--)
#define FOR(i, j, n) for (ll i = j; i < n; i++)
#define ig cin.ignore()
#define fast_cin ios_base::sync_with_stdio(false)
// The following helper functions are unused in this specific problem.
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
ll inv(ll x) { return power(x, mod - 2); }
#define run(m) for (auto it = m.begin(); it != m.end(); it++)
/*---------------------------------------------------------//
const int MAXM = 1e7+3010;   ll fact[MAXM];
void factorial(){ fact[0]=1;FOR(i,1,MAXM) fact[i]=(fact[i-1]*i)%mod;}
ll nCr(ll n,ll r){return
((fact[n])*(((inv(fact[r]))*(inv(fact[n-r])))%mod))%mod;}
// freopen("Task.in","r",stdin);freopen("Task.out","w",stdout);
//-------------------------------------------------------------------/*/ 

int main() {
  fast_cin;

  ll n, t;
  cin >> n >> t;
  string s;
  cin >> s;

  /**
   * @brief Outer loop to simulate `t` time steps.
   */
  while (t--) {

    /**
     * @brief Inner loop to perform swaps for a single time step.
     * @details This loop scans the queue and swaps every adjacent 'BG' pair.
     * Incrementing `i` after the swap is crucial as it moves the scan forward,
     * effectively ensuring that a child moves at most once per time step.
     */
    FOR(i, 0, n - 1) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        i++; // Move past the swapped pair.
      }
    }
  }
  cout << s << "\n";
}