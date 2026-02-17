/**
 * @file 1426B.CPP
 * @brief Solution for a matrix construction problem, likely Codeforces 1426B.
 * @details This program determines if it's possible to construct a symmetric m x m
 * matrix using a given set of n types of 2x2 tiles.
 *
 * @section
 *
 * Algorithm:
 * The solution is based on identifying two necessary conditions for constructing the
 * symmetric matrix:
 * 1. The dimension of the target matrix, 'm', must be even. This is because the
 *    building blocks are 2x2 tiles, which can only tile a grid with even dimensions.
 * 2. There must be at least one type of 2x2 tile that is itself symmetric. A tile
 *    [[a, b], [c, d]] is symmetric if b == c. Such a tile is required for placement
 *    along the main diagonal of the larger m x m matrix to maintain overall symmetry.
 *
 * The code checks for the existence of at least one symmetric tile among the 'n'
 * types provided. If a symmetric tile is found and 'm' is even, it's possible ("YES").
 * Otherwise, it's not possible ("NO").
 *
 * - Time Complexity: O(T * N) for T test cases, each taking O(N) to iterate through the tiles.
 * - Space Complexity: O(1) per test case.
 *
 * Note: The file contains extensive boilerplate macros and helper functions common in
 * competitive programming, which are not used in the core logic for this problem.
 */
#include <bits/stdc++.h>
#include <iostream>
using namespace std;

#define rep(i, a, b) for (int i = a; i < b; i++)
#define repe(i, a, b) for (int i = a; i <= b; i++)
#define ll long long
#define ui unsigned int
#define pb push_back
#define deb(x) cout << #x << '=' << x << endl
#define deb2(x, y) cout << #x << '=' << x < < < < #y << '=' << y << endl
#define all(x) x.begin(), x.end()
#define endl "\n"
#define ff first
#define ss second
#define lb lower_bound
#define ub upper_bound
#define INF (int)1e9
#define PI 3.1415926535897932384626433832795
#define MOD 1000000007

#define hcf(x, y) __gcd(x, y)
#define lcm(x, y) (x * y) / (__gcd(x, y))
#define CountOne(x) __builtin_popcount(x)
#define parity(x)                                                              \
  __builtin_parity(x) // Funtion return (true) if number of set bits is odd(odd
                      // parity) else false..
#define LeadZero(x)                                                            \
  __builtin_clz(x) // Counts leading zero in binary representation of x ....
#define TrailZero(x) __builtin_ctz(x)

#define fastIO                                                                 \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0)

typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<ll> vl;
typedef vector<vl> vvl;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef pair<string, string> pss;
typedef map<int, int> mii;
typedef unordered_map<int, int> umap_ii;
typedef unordered_map<int, int> umap_ii;
typedef unordered_map<string, int> umap_si;
mt19937_64
    rang(chrono::high_resolution_clock::now().time_since_epoch().count());
// alt + ctrl + N

//===============================================================================//
ll power(ll x, ll y) {
  ll v = 1;
  while (y > 0) {
    if (y & 1)
      v = v * x;
    y = y >> 1;
    x = x * x;
  }
  return v;
}

bool isPowerOfTwo(ll x) {
  return x && (!(x & (x - 1))); // check if the number is power of two or not
}

//===============================================================================//

int main() {
  fastIO;
  srand(chrono::high_resolution_clock::now().time_since_epoch().count());

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  ll test = 1;
  cin >> test;
  /**
   * Block Logic: Main loop to handle multiple test cases.
   */
  while (test--) {
    ll n, m;
    cin >> n >> m;
    ll check = 0; // Flag to indicate if a symmetric tile is found.
    /**
     * Block Logic: Iterates through the 'n' available tile types to find a symmetric one.
     * Invariant: 'check' is 1 if a symmetric tile has been found in the preceding iterations.
     */
    rep(i, 0, n) {
      ll a, b, c, d;
      cin >> a >> b >> c >> d; // Reads a 2x2 tile: [[a, b], [c, d]]
      /**
       * Block Logic: Checks if the current tile is symmetric (b == c).
       * Such a tile can be placed on the main diagonal of the target matrix.
       */
      if (b == c) {
        check = 1;
      }
    }
    /**
     * Block Logic: A symmetric m x m matrix cannot be tiled perfectly with 2x2 tiles if 'm' is odd.
     * This resets the possibility if 'm' is not an even number.
     */
    if (m % 2 == 1)
      check = 0;
    
    // Output "YES" if a symmetric tile was found AND m is even. Otherwise, "NO".
    if (check)
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  }
  return 0;
}
