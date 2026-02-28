/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem from Codeforces (266A).
 * @details The problem requires finding the minimum number of stones to remove from a row
 * such that no two adjacent stones have the same color. This file also contains a
 * number of unused utility functions common in competitive programming.
 *
 * @algorithm
 * The core logic resides in the `solve()` function. It iterates through the sequence of stones
 * (represented by a string) and counts the number of times two adjacent stones share the
 * same color. This count is precisely the minimum number of stones that must be
 * removed to satisfy the problem's condition.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of stones, due to the single pass through the string.
 * Space Complexity: O(N) to store the string representing the stone colors.
 */
#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define N 1000005
#define M 1000000007

/*
 *********************************************************************
 *                             Code By                               *
 *                                                                   *
 *                            Srajan Gupta                           *
 *                            srajang_123                            *
 *                                                                   *
 *********************************************************************
 */
// Unused competitive programming template functions follow.
// Prime Numbers

vector<bool> prime(N + 1, true);
void sieve() {
  ll i, j, k;
  prime[0] = prime[1] = false;
  for (i = 2; i * i <= N; i++) {
    if (prime[i]) {
      for (j = i * i; j <= N; j += i) {
        prime[j] = false;
      }
    }
  }
}

// Exponentiation

ll power(ll a, ll b) {
  ll r = 1;
  while (b) {
    if (b % 2 == 1)
      r = r * a;
    b /= 2;
    a *= a;
  }
  return r;
}
ll power(ll a, ll b, ll m) {
  a = a % m;
  ll r = 1;
  while (b) {
    if (b % 2 == 1)
      r = (r * a) % m;
    b /= 2;
    a = (a * a) % m;
  }
  return r;
}

// Prime Factors

vector<ll> factors(ll n) {
  vector<ll> r;
  ll i, j;
  for (i = 1; i * i <= n; i++) {
    if (n % i == 0) {
      r.push_back(i);
      if (n / i != i)
        r.push_back(n / i);
    }
  }
  return r;
}

// GCD
ll gcd(ll a, ll b) {
  if (b > a) {
    ll t = a;
    a = b;
    b = t;
  }
  if (b == 0)
    return a;
  return gcd(b, a % b);
}
ll lcm(ll a, ll b) { return (a * b) / gcd(a, b); }
// Graphs
vector<bool> bvisited(N, false);
vector<vector<ll>> G(N);
vector<ll> bfs(ll s) {
  vector<ll> order;
  queue<ll> q;
  bvisited[s] = true;
  q.push(s);
  while (!q.empty()) {
    s = q.front();
    q.pop();
    order.push_back(s);
    for (auto x : G[s]) {
      if (!bvisited[x]) {
        bvisited[x] = true;
        q.push(x);
      }
    }
  }
  return order;
}
vector<bool> dvisited(N, false);
vector<ll> dfs(ll s) {
  vector<ll> order;
  stack<ll> q;
  q.push(s);
  while (!q.empty()) {
    ll v = q.top();
    q.pop();
    if (!dvisited[v]) {
      order.push_back(v);
      dvisited[v] = true;
    }
    for (ll i = 0; i < G[v].size(); i++) {
      if (!dvisited[G[v][i]]) {
        q.push(G[v][i]);
      }
    }
  }
  return order;
}
// My Functions

void print(pair<ll, ll> a) { cout << a.first << " " << a.second; }
void print(vector<ll> a) {
  for (auto x : a)
    cout << x << " ";
}
bool sortbysec(const pair<ll, ll> &a, const pair<ll, ll> &b) {
  return a.second < b.second;
}
// Main Solution for Codeforces Problem 266A

void solve() {
  /**
   * @variable n: The number of stones.
   * @variable s: A string representing the colors of the stones.
   * @variable k: A counter for the minimum number of stones to remove.
   */
  ll n, i, j, k = 0;
  string s;
  cin >> n >> s;
  /**
   * @block
   * @brief This loop iterates through the stones to find adjacent pairs of the same color.
   * @invariant `k` holds the count of removals needed so far.
   * The loop starts from the second stone (index 1) to compare it with its predecessor.
   */
  for (i = 1; i < n; i++) {
    // Pre-condition: Check if the current stone has the same color as the previous one.
    if (s[i] == s[i - 1])
      k++; // If they are the same, one must be removed.
  }
  // Output the final count.
  cout << k;
}

int main() {
  // Fast I/O setup
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  ll t = 1;
  // Number of test cases, fixed to 1 for this problem.
  // cin>>t;
  while (t--) {
    solve();
  }
}
