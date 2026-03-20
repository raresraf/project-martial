/**
 * @file 546A.CPP
 * @brief Solution to Codeforces problem 546A - Soldier and Bananas.
 *        This program calculates the total cost to buy a certain number of bananas,
 *        given that the price of each subsequent banana increases.
 *        It then determines how much money the soldier needs to borrow if his initial
 *        money is insufficient.
 * Algorithm:
 *   1. Read the initial price of the first banana (k), the soldier's initial money (n),
 *      and the number of bananas he wants to buy (w).
 *   2. Calculate the sum of an arithmetic progression: k * 1 + k * 2 + ... + k * w.
 *      This represents the total cost.
 *   3. If the total cost is less than or equal to the soldier's initial money, no money
 *      needs to be borrowed.
 *   4. Otherwise, calculate the difference between the total cost and the initial money
 *      to find the amount to borrow.
 * Time Complexity: O(w) for the loop to calculate the sum, where w is the number of bananas.
 *                  Alternatively, O(1) using the arithmetic series sum formula (w * (w+1) / 2 * k).
 *                  The current implementation uses a loop.
 * Space Complexity: O(1) as only a few variables are used.
 */
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
int mpow(int base, int exp);
void ipgraph(int m);
void dfs(int u, int par);
const int mod = 1000000007;
const int N = 3e5, M = N;
//=======================

vi g[N];
int a[N];

int main() {
  /**
   * @brief Main function implementing the solution for the "Soldier and Bananas" problem.
   *
   * Reads the input values (initial banana price 'k', initial money 'n', number of bananas 'w').
   * Calculates the total cost of buying 'w' bananas, where the cost of each banana
   * increases arithmetically. Determines and prints the amount of money the soldier needs to borrow.
   * This function directly solves the problem logic.
   * @return int Returns 0 upon successful execution.
   */
  int n, k, w;
  int i;
  cin >> k >> n >> w;
  int sum = 0;
  // Block Logic: Calculates the total cost of buying 'w' bananas.
  // The price of each banana increases arithmetically: 1*k, 2*k, ..., w*k.
  // Invariant: 'sum' accumulates the total cost.
  fo(i, w) { sum = sum + k * i; }
  // Block Logic: Determines the amount of money the soldier needs to borrow.
  // Pre-condition: 'sum' is the total cost, 'n' is the soldier's initial money.
  // Post-condition: Prints 0 if 'n' is sufficient, otherwise prints the deficit.
  if (sum < n)
    cout << 0;
  else
    cout << sum - n;
  return 0;
}

int mpow(int base, int exp) {
  /**
   * @brief Calculates (base^exp) % mod using binary exponentiation (exponentiation by squaring).
   * @param base The base of the exponentiation.
   * @param exp The exponent.
   * @return int The result of (base^exp) % mod.
   * @complexity Time Complexity: O(log exp)
   */
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
  /**
   * @brief Reads graph input and constructs an adjacency list representation.
   *
   * This function is typically used to build an undirected graph where edges are
   * provided as pairs of vertices (u, v). It assumes 0-indexed vertices
   * after conversion from 1-indexed input.
   * @param n The number of vertices in the graph (though not directly used in the loop, implies graph size).
   * @param m The number of edges to read.
   */
  int i, u, v;
  while (m--) {
    cin >> u >> v;
    g[u - 1].pb(v - 1);
    g[v - 1].pb(u - 1);
  }
}

void dfs(int u, int par) {
  /**
   * @brief Performs a Depth First Search (DFS) traversal on a graph.
   *
   * This is a standard recursive DFS implementation that visits all reachable
   * nodes from a starting node 'u', avoiding revisiting the immediate parent 'par'.
   * @param u The current node being visited.
   * @param par The parent node of 'u' in the DFS tree, used to prevent going backwards.
   */
  for (int v : g[u]) {
    if (v == par)
      continue;
    dfs(v, u);
  }
}
