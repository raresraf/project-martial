/**
 * @file 112A.CPP
 * @brief Solution to the "Petya and Strings" problem, performing case-insensitive string comparison.
 * @details This program compares two strings lexicographically, ignoring the case of the characters.
 * It uses the C++ Standard Library's `transform` function for efficient case conversion and `std::string`'s
 * built-in comparison operators. The file also contains a large template of unused macros and functions
 * common in competitive programming.
 *
 * Algorithm:
 * 1. Read two strings, `a` and `b`.
 * 2. Use `std::transform` to convert both strings to their lowercase equivalents in-place.
 * 3. Directly compare the modified strings using the overloaded operators for `std::string`.
 *    - If `a` is lexicographically less than `b`, print -1.
 *    - If `a` is equal to `b`, print 0.
 *    - If `a` is lexicographically greater than `b`, print 1.
 *
 * Time Complexity: O(L), where L is the length of the strings. This is dominated by the `transform` and comparison operations.
 * Space Complexity: O(1), as the strings are modified in-place and no significant extra storage is used.
 */
#include <bits/stdc++.h>
using namespace std;
// The following is a common template for competitive programming, containing many unused macros and functions for this specific problem.
#define gc getchar_unlocked
#define fo(i, n) for (i = 0; i < n; i++)
#define Fo(i, k, n) for (i = k; k < n ? i < n : i > n; k < n ? i += 1 : i -= 1)
#define ll long long
#define si(x) scanf("%d", &x)
#define sl(x) scanf("%lld", &x)
#define ss(s) scanf("%s", s)
#define pi(x) printf("%d\n", x)
#define pl(x) printf("%lld\n", x)
#define ps(s) printf("%s\n", s)
#define deb(x) cout << #x << "=" << x << endl
#define deb2(x, y) cout << #x << "=" << x << "," << #y << "=" << y << endl
#define pb push_back
#define mp make_pair
#define F first
#define S second
#define all(x) x.begin(), x.end()
#define clr(x) memset(x, 0, sizeof(x))
#define sortall(x) sort(all(x))
#define tr(it, a) for (auto it = a.begin(); it != a.end(); it++)
#define PI 3.1415926535897932384626
typedef pair<int, int> pii;
typedef pair<ll, ll> pl;
typedef vector<int> vi;
typedef vector<ll> vl;
typedef vector<pii> vpii;
typedef vector<pl> vpl;
typedef vector<vi> vvi;
typedef vector<vl> vvl;

// Unused template functions for modular exponentiation, graph input, and DFS.
int mpow(int base, int exp);
void ipgraph(int m);
void dfs(int u, int par);

const int mod = 1000000007;
const int N = 3e5, M = N;
//=======================

vi g[N];
int a[N];

int main() {
  // Optimizes C++ standard streams for faster I/O.
  ios_base::sync_with_stdio(0);
  ll t, i, j, k, p, q, r, x, y, u, v, n, m; // Many unused variables from the template.
  string a, b;
  cin >> a >> b;

  // Block Logic: Convert both strings to lowercase for case-insensitive comparison.
  transform(a.begin(), a.end(), a.begin(), ::tolower);
  transform(b.begin(), b.end(), b.begin(), ::tolower);
  
  /**
   * Block Logic: Perform lexicographical comparison on the normalized strings.
   * std::string's comparison operators are used for a direct and efficient comparison.
   */
  if (a < b)
    cout << "-1" << endl;
  else if (a == b)
    cout << "0" << endl;
  else
    cout << "1" << endl;

  return 0;
}

// Unused template function: Modular Exponentiation.
int mpow(int base, int exp) {
  base %= mod;
  int result = 1;
  while (exp > 0) {
    if (exp & 1)
      result = ((ll)result * base) % mod;
    base = ((ll)base * base) % mod;
    exp >>= 1;
  }
  return result;
}

// Unused template function: Input graph for adjacency list representation.
void ipgraph(int n, int m) {
  int i, u, v;
  while (m--) {
    cin >> u >> v;
    g[u - 1].pb(v - 1);
    g[v - 1].pb(u - 1);
  }
}

// Unused template function: Depth First Search.
void dfs(int u, int par) {
  for (int v : g[u]) {
    if (v == par)
      continue;
    dfs(v, u);
  }
}
