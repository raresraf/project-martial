/**
 * @file 281A.CPP
 * @brief A competitive programming template that solves the "Word Capitalization" problem.
 * @details This file contains a large amount of boilerplate code for competitive
 * programming, such as macros, type definitions, and helper functions for various
 * domains (math, graphs). The actual solution to the specific problem is
 * contained within the `solve()` function.
 */
// Problem Link : https://codeforces.com/problemset/problem/281/A

#include <bits/stdc++.h>

using namespace std;

// Note: Most of the following macros and functions are part of a generic
// template and are not used in this specific problem's solution.
#define gc getchar_unlocked
#define rep(i, a, b) for (ll i = (a); i < (b); i++)
#define rrep(i, a, b) for (ll i = (a); i > (b); i--)
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
#define to_lc(s)                                                               \
  transform(all(s), s.begin(),                                                 \
            ::tolower); // convert string(STL) passed to lowercase
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

mt19937_64
    rang(chrono::high_resolution_clock::now().time_since_epoch().count());
int rng(int lim) {
  uniform_int_distribution<int> uid(0, lim - 1);
  return uid(rang);
}

clock_t startTime;
double getCurrentTime() {
  return (double)(clock() - startTime) / CLOCKS_PER_SEC;
}

const int inf = 0x3fffffff;
const ll INF = 0x1fffffffffffffff;
const ll mod = 1e9 + 7;

ll mpow(ll base, ll exp) {
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

// Graph...
const int N = 3e5, M = N;
vi g[N];

void ipgraph(int n, int m) {
  int i, u, v;
  while (m--) {
    cin >> u >> v;
    u--, v--;
    g[u].pb(v);
    g[v].pb(u);
  }
}

void dfs(int u, int par) {
  for (int v : g[u]) {
    if (v == par)
      continue;
    dfs(v, u);
  }
}

/**
 * @brief Solves the "Word Capitalization" problem (Codeforces 281A).
 * @details The function reads a single word and capitalizes its first letter,
 * leaving the rest of the word unchanged.
 *
 * @algorithm
 * 1. Read the input word into a string `s`.
 * 2. Use `std::transform` to apply the `::toupper` function specifically to the
 *    first character of the string (range `s.begin()` to `s.begin() + 1`).
 * 3. Print the modified string.
 *
 * @complexity Time: O(L) for reading the string, O(1) for capitalization.
 *             Space: O(L) for storing the string.
 */
void solve() {
  string s;
  cin >> s;
  // Apply toupper to only the first character of the string.
  transform(s.begin(), s.begin() + 1, s.begin(), ::toupper);
  cout << s << endl;
}

int main() {
  // Fast I/O setup and random number generator seeding.
  ios_base::sync_with_stdio(0), cin.tie(0), cout.tie(0);
  srand(chrono::high_resolution_clock::now().time_since_epoch().count());

  startTime = clock();

// Redirects I/O to files for local testing when not compiling online.
#ifndef ONLINE_JUDGE
  // for getting input from input.txt
  freopen("input.txt", "r", stdin);
  // for writing output to output.txt
  freopen("output.txt", "w", stdout);
#endif

  // Standard template loop for multiple test cases, hardcoded to one here.
  ll t = 1;
  // cin >> t;
  while (t--) {
    solve();
  }

  return 0;
}
