/**
 * @file 50A.CPP
 * @brief Logic for optimal tiling of a rectangular board with dominoes.
 * 
 * Logic: Calculates the maximum number of 2x1 dominoes that can fit on an MxN grid 
 * without overlap or overflow. Since each domino occupies exactly 2 units of area, 
 * the maximum count is the integer division of the total area by 2.
 * 
 * Algorithm: Constant-time area-based calculation.
 * Time Complexity: $O(1)$ constant time.
 * Space Complexity: $O(1)$ constant space.
 */

#pragma GCC optimize("-O3")

#include <bits/stdc++.h>
#include <cmath>

using namespace std;

typedef long long int ll;
typedef long double ld;
typedef pair<int, int> p32;
typedef pair<ll, ll> p64;
typedef pair<double, double> pdd;
typedef vector<ll> v64;
typedef vector<int> v32;
typedef vector<vector<int>> vv32;
typedef vector<vector<ll>> vv64;
typedef vector<p64> vp64;
typedef vector<p32> vp32;

ll MOD = 998244353;
ll NUM = 1e9 + 7;

#define forn(i, e) for (ll i = 0; i < e; i++)
#define forsn(i, s, e) for (ll i = s; i < e; i++)
#define rforn(i, s) for (ll i = s; i >= 0; i--)
#define rforsn(i, s, e) for (ll i = s; i >= e; i--)
#define ln "\n"
#define dbg(x) cout << #x << " = " << x << ln
#define mp make_pair
#define pb push_back
#define ff first
#define ss second
#define INF 1e18

/**
 * Functional Utility: Optimizes standard I/O for high-performance stream processing.
 */
#define fast_cin()                                                             \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL)

#define all(x) (x).begin(), (x).end()
#define sz(x) ((ll)(x).size())
#define zer ll(0)

/**
 * @brief core solver for the domino tiling problem.
 */
void solve() {
  int m, n;
  // Functional Utility: Reads grid dimensions.
  cin >> m >> n;

  /**
   * Logic: Max dominoes = floor(Total Area / Area of 1 domino).
   * Note: This holds true regardless of orientation because a 2x1 domino can 
   * always perfectly cover any two adjacent cells.
   */
  cout << (m * n) / 2 << ln;
}

int main() {
  fast_cin();

  ll t = 1;
  // cin>>t; // Optional multi-case support.

  /**
   * Block Logic: Orchestrates the execution of the solver.
   */
  while (t--)
    solve();

  return 0;
}
