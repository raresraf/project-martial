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
/**
 * @58103302-6117-4f0b-a9b6-e25769b73594/266A.CPP
 * @brief This program primarily solves problem 266A from competitive programming,
 * which involves counting adjacent identical characters in an input string.
 * It also includes a collection of common competitive programming utility functions
 * (e.g., sieve for primes, power, factors, GCD, graph traversals like BFS/DFS, and print utilities)
 * which are available but not directly utilized by the `solve()` function for this specific problem.
 *
 * The core logic for problem 266A is encapsulated within the `solve()` function.
 *
 * Algorithm (solve function): Single pass linear scan to count adjacent identical characters.
 * Time Complexity (solve function): O(N) where N is the length of the input string.
 * Space Complexity (solve function): O(N) for string 's' and O(1) for other variables.
 */
#include <bits/stdc++.h> // Includes most standard C++ library headers.
using namespace std;     // Uses the standard namespace.
#define ll long long     // Defines 'll' as a shorthand for long long.
#define N 1000005        // Defines N for a common maximum array/container size.
#define M 1000000007     // Defines M for a common modulo value.

// Prime Numbers (Utility Function Block - not directly used by solve for 266A)

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

// Exponentiation (Utility Function Block - not directly used by solve for 266A)

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

// Prime Factors (Utility Function Block - not directly used by solve for 266A)

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

// GCD & LCM (Utility Function Block - not directly used by solve for 266A)
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
// Graphs (Utility Function Block - not directly used by solve for 266A)
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
// My Functions (Utility Function Block - not directly used by solve for 266A)

void print(pair<ll, ll> a) { cout << a.first << " " << a.second; }
void print(vector<ll> a) {
  for (auto x : a)
    cout << x << " ";
}
bool sortbysec(const pair<ll, ll> &a, const pair<ll, ll> &b) {
  return a.second < b.second;
}
// Main Solution
/**
 * @brief Solves problem 266A: counts adjacent identical characters in a string.
 *
 * This function reads an integer 'n' and a string 's' of length 'n'.
 * It then iterates through the string, comparing each character with its preceding
 * character. If they are identical, a counter 'k' is incremented.
 * Finally, the total count 'k' is printed.
 *
 * Algorithm: Linear scan.
 * Time Complexity: O(n)
 * Space Complexity: O(1) (excluding input string storage).
 */
void solve() {
  ll n, i, j, k = 0; // Declares n (string length), i, j (loop counters), and k (adjacent identical char counter).
  string s;          // Declares a string variable to store the input sequence.
  cin >> n >> s;     // Reads the length 'n' and the string 's' from standard input.
  /**
   * Block Logic: Iterates through the string 's' from the second character (index 1)
   * to compare each character with its immediate predecessor.
   * Invariant: 'k' maintains the count of adjacent identical characters found so far.
   */
  for (i = 1; i < n; i++) { // Loop starts from the second character to allow comparison with s[i-1].
    // Inline: Checks if the current character s[i] is the same as the previous character s[i-1].
    if (s[i] == s[i - 1]) 
      k++; // Increments 'k' if an adjacent identical pair is found.
  }
  cout << k; // Prints the final count of adjacent identical characters.
}

int main() {
  // Optimizes C++ standard I/O streams for competitive programming.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  ll t = 1; // Initializes 't' to 1; usually represents number of test cases.
  // cin>>t; // Commented out, so it runs for a single test case.
  /**
   * Block Logic: Calls the 'solve' function for each test case.
   * Invariant: The 'solve' function is executed 't' times.
   */
  while (t--) { 
    solve(); // Calls the function containing the solution logic for problem 266A.
  }
}
