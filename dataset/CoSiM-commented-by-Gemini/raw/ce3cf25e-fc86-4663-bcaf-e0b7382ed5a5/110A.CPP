/**
 * @file 110A.CPP
 * @brief Solution for competitive programming problem "Nearly Lucky Number".
 * @details This file contains a solution to a problem that checks if a number
 *          is "nearly lucky". A number is nearly lucky if the count of its
 *          lucky digits ('4' and '7') is itself a lucky number (a positive
 *          integer composed only of '4's and '7's).
 *
 *          This file also includes a large template of common utility functions
 *          for competitive programming, most of which are not used for this
 *          specific problem.
 *
 * Time Complexity (Solve): O(log(n))
 * Space Complexity (Solve): O(1)
 */
#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define N 1000005
#define M 1000000007

// --- Start of Competitive Programming Template ---

// Prime Numbers
vector<bool> prime(N + 1, true);
/**
 * @brief Populates a boolean vector indicating primality up to N using the Sieve of Eratosthenes.
 * @algo Sieve of Eratosthenes
 * @complexity Time: O(N log log N), Space: O(N)
 */
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
/**
 * @brief Calculates (a^b) using binary exponentiation.
 * @complexity Time: O(log b)
 */
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
/**
 * @brief Calculates (a^b) % m using binary exponentiation.
 * @complexity Time: O(log b)
 */
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
/**
 * @brief Finds all factors (divisors) of a given number n.
 * @return A vector containing all factors of n.
 * @complexity Time: O(sqrt(n))
 */
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
/**
 * @brief Computes the greatest common divisor of two numbers using the Euclidean algorithm.
 * @complexity Time: O(log(min(a,b)))
 */
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
/**
 * @brief Computes the least common multiple of two numbers.
 */
ll lcm(ll a, ll b) { return (a * b) / gcd(a, b); }

// Graphs
vector<bool> bvisited(N, false);
/**
 * @brief Performs a Breadth-First Search on a graph.
 * @param V Number of vertices.
 * @param G Adjacency list representation of the graph.
 * @param s Starting vertex.
 * @return A vector with nodes in BFS traversal order.
 */
vector<ll> bfs(ll V, vector<vector<ll>> G, ll s) {
  vector<ll> order;
  queue<ll> q;
  bvisited[s] = true;
  q.push(s);
  while (!q.empty()) {
    s = q.front();
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
/**
 * @brief Performs a Depth-First Search on a graph (iterative version).
 * @param V Number of vertices.
 * @param G Adjacency list representation of the graph.
 * @param s Starting vertex.
 * @return A vector with nodes in DFS traversal order.
 */
vector<ll> dfs(ll V, vector<vector<ll>> G, ll s) {
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

// --- End of Template ---

/**
 * @brief Contains the core logic for the "Nearly Lucky Number" problem.
 */
void solve() {
  ll n, i, j, k, l = 0;
  cin >> n;

  // First pass: Count the number of lucky digits ('4' or '7') in the input `n`.
  // The count is stored in 'l'.
  while (n) {
    if (n % 10 == 4 || n % 10 == 7)
      l++;
    n /= 10;
  }

  k = 0, j = 0;
  ll u = l; // Keep a copy of the original count to check for positivity.

  // Second pass: Check if the count 'l' is itself a lucky number.
  // 'j' counts the total digits in 'l'.
  // 'k' counts the number of lucky digits in 'l'.
  while (l) {
    j++;
    if (l % 10 == 4 || l % 10 == 7)
      k++;
    l /= 10;
  }

  // A number is nearly lucky if the count of its lucky digits (`u`) is positive
  // AND all digits of the count are lucky digits (`j == k`).
  if (j == k && u > 0)
    cout << "YES";
  else
    cout << "NO";
}

int main() {
  // Fast I/O setup.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  
  // Standard template for multiple test cases, though here t=1.
  ll t = 1;
  while (t--) {
    solve();
  }
}
