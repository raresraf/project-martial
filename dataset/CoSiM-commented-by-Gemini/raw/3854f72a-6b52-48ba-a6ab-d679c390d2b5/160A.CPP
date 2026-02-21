/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem by finding the minimum number of coins to take to have a sum greater than the remaining coins.
 *
 * This program determines the smallest number of coins a person must take from a set
 * so that the sum of their chosen coins is strictly greater than the sum of the
 * remaining coins. The strategy employed is a greedy one.
 *
 * Algorithm:
 * 1. Read all coin values into a vector.
 * 2. Calculate the total sum of all coin values.
 * 3. Sort the coins in descending order.
 * 4. Iterate through the sorted coins, adding their values to a running "my_sum".
 * 5. In each iteration, check if "my_sum" is greater than the sum of the remaining coins
 *    (total_sum - my_sum).
 * 6. The number of coins taken when this condition is first met is the answer.
 *
 * Time Complexity: O(N log N), dominated by the sorting of the coins.
 * Space Complexity: O(N) to store the coin values.
 *
 * Note: This file contains a large number of macros and function definitions
 * (e.g., `mpow`, `ipgraph`, `dfs`) that are part of a generic competitive
 * programming template and are not used in this specific solution.
 */
#include <bits/stdc++.h>
using namespace std;
// Unused competitive programming macros and template functions.
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
  
  // Sort the coins in descending order to apply a greedy strategy.
  sort(ar.begin(), ar.end(), greater<int>());
  
  ll sum2 = 0; // This will be the sum of coins taken.
  i = 0;
  /**
   * @brief Greedily take the largest coins until their sum is more than half the total.
   * Pre-condition: `ar` is sorted in descending order. `sum` holds the total value.
   * Invariant: `sum2` is the sum of the `i` largest coins.
   * The loop continues as long as the sum of coins taken (`sum2`) is not strictly
   * greater than the sum of the remaining coins (`sum - sum2`).
   */
  while (i < ar.size() && sum2 <= sum - sum2) {
    sum2 += ar[i++];
  }
  
  // Output the minimum number of coins needed.
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
