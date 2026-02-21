/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem (Codeforces 160A) using a greedy approach.
 *
 * @details The problem is to find the minimum number of coins to take from a set
 * so that the sum of the taken coins is strictly greater than the sum of the
 * remaining coins. This program implements a greedy algorithm to solve this.
 *
 * @algorithm The core of the solution is a greedy strategy.
 * 1. Calculate the total sum of all coin values.
 * 2. Sort the coins in descending order.
 * 3. Iterate through the sorted coins (largest to smallest), accumulating their
 *    sum. Stop when this accumulated sum is strictly greater than the sum of
 *    the remaining coins (which is `total_sum - accumulated_sum`).
 * 4. The count of coins taken is the minimum required.
 *
 * @complexity
 * Time Complexity: O(N log N), where N is the number of coins. The sorting
 *                  operation is the performance bottleneck.
 * Space Complexity: O(N), to store the values of the N coins.
 */
#include <bits/stdc++.h>
using namespace std;
// NOTE: The following section contains a large number of macros and unused
// function declarations, typical of a competitive programming template.
// The core logic is within the main function.
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
// Unused template functions for modular exponentiation and graph processing.
int mpow(int base, int exp);
void ipgraph(int m);
void dfs(int u, int par);
const int mod = 1000000007;
const int N = 3e5, M = N;
//=======================

vi g[N];
int a[N];

int main() {
  int n, i;
  cin >> n;
  vi ar(n);
  ll sum = 0;
  // Read coin values and calculate the total sum.
  fo(i, n) {
    cin >> ar[i];
    sum += ar[i];
  }
  // Sort coins in descending order to apply the greedy strategy.
  sort(ar.begin(), ar.end(), greater<int>());
  
  ll sum2 = 0; // This will be the sum of coins taken.
  i = 0;
  /**
   * @brief Block Logic: Greedily take the largest coins until their sum is
   * greater than the sum of the remaining coins.
   * @invariant `sum2` is the sum of the `i` largest coins.
   * The loop condition `sum2 <= sum - sum2` checks if our share is not yet
   * strictly greater than the other share.
   */
  while (i <= ar.size() and sum2 <= sum - sum2) {
    sum2 += ar[i++];
  }
  // Output the minimum number of coins required.
  cout << i;

  return 0;
}

// Unused template function for modular exponentiation.
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

// Unused template function for graph input.
void ipgraph(int n, int m) {
  int i, u, v;
  while (m--) {
    cin >> u >> v;
    g[u - 1].pb(v - 1);
    g[v - 1].pb(u - 1);
  }
}

// Unused template function for Depth First Search.
void dfs(int u, int par) {
  for (int v : g[u]) {
    if (v == par)
      continue;
    dfs(v, u);
  }
}
