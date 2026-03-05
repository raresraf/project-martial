/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem using a loop-based summation.
 *
 * This program calculates the amount of money a soldier needs to borrow to buy `w`
 * bananas, where the costs increase linearly (`k`, `2k`, `3k`, ...).
 *
 * NOTE: This file contains a large amount of boilerplate template code for competitive
 * programming, including functions for modular exponentiation, graph input, and DFS,
 * which are all unused for this specific problem.
 *
 * @algorithm
 * This implementation calculates the total cost by directly simulating the sum,
 * rather than using the closed-form formula for an arithmetic series.
 *
 * 1. Read the input values: `k` (base cost), `n` (initial money), and `w`
 *    (number of bananas).
 * 2. Initialize a `sum` variable to 0.
 * 3. Loop from `i = 1` to `w`. In each iteration, `i` represents the `i`-th banana.
 *    a. Calculate the cost of the `i`-th banana, which is `k * i`.
 *    b. Add this cost to the running `sum`.
 * 4. After the loop, `sum` holds the total cost of all `w` bananas.
 * 5. Compare the `total` cost with the soldier's money `n`.
 *    a. If `sum < n` (or `sum <= n`), the soldier has enough money, and the borrowed
 *       amount is 0.
 *    b. Otherwise, the amount to borrow is the difference, `sum - n`.
 * 6. Print the result.
 *
 * @complexity
 * Time Complexity: O(w), due to the loop that runs `w` times to calculate the sum.
 * This is less efficient than the O(1) formula-based approach but sufficient for
 * the problem's constraints.
 * Space Complexity: O(1).
 */

// --- Boilerplate and Unused Code Section ---
#include <bits/stdc++.h>
using namespace std;
#define gc getchar_unlocked
#define fo(i, n) for (i = 1; i <= n; i++)
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
// Unused function prototypes and global variables.
int mpow(int base, int exp);
void ipgraph(int m);
void dfs(int u, int par);
const int mod = 1000000007;
const int N = 3e5, M = N;
vi g[N];
int a[N];
// --- End of Boilerplate Section ---

int main() {
  int n, k, w;
  int i;
  cin >> k >> n >> w;

  // `sum` will accumulate the total cost. Declared as `int`, which is sufficient
  // for the problem constraints but `long long` would be safer practice.
  int sum = 0;

  /**
   * @brief Calculate the total cost by iterating from the 1st to the w-th banana.
   * The custom `fo` macro is equivalent to `for (i = 1; i <= w; i++)`.
   */
  fo(i, w) {
    sum = sum + k * i;
  }
  
  // Check if the soldier has enough money.
  if (sum <= n) // Corrected from `sum < n` to handle the case where cost equals money.
    cout << 0;
  else
    // Print the amount to borrow.
    cout << sum - n;

  return 0;
}

// --- Unused Template Functions ---
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

void ipgraph(int n, int m) {
  int i, u, v;
  while (m--) {
    cin >> u >> v;
    g[u - 1].pb(v - 1);
    g[v - 1].pb(u - 1);
  }
}

void dfs(int u, int par) {
  for (int v : g[u]) {
    if (v == par)
      continue;
    dfs(v, u);
  }
}
