/**
 * @file 110A.CPP
 * @brief This file is a common competitive programming template containing various utility functions
 * and a solution to the "Nearly Lucky Number" problem (Codeforces 110A).
 * The problem determines if a given number is "nearly lucky", which means the count
 * of lucky digits (4 and 7) within the number is itself a lucky number.
 * Algorithm: Digit extraction, counting, and recursive lucky number check.
 * Time Complexity: O(log10(N)) for the `solve` function, where N is the input number.
 * The overall complexity is dominated by the template utility functions if used, e.g., O(N log log N) for sieve.
 * Space Complexity: O(N) for sieve, O(log10(N)) for `solve` (if considering stack space for recursion in general).
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries (e.g., iostream, vector, algorithm).
using namespace std;

// Defines 'll' as a shorthand for 'long long' for convenience in competitive programming.
#define ll long long
// Defines 'N' as a maximum array/vector size or limit, often used for sieve or array declarations.
#define N 1000005
// Defines 'M' as a modulo constant, typically used for operations in modular arithmetic.
#define M 1000000007

// Prime Numbers
// Declares a boolean vector 'prime' of size N+1, initialized to true, for sieve of Eratosthenes.
vector<bool> prime(N + 1, true);

/**
 * @brief Implements the Sieve of Eratosthenes to find prime numbers up to N.
 *
 * This function marks all non-prime numbers up to N by setting their corresponding
 * entry in the 'prime' boolean vector to false.
 * Time Complexity: O(N log log N).
 * Space Complexity: O(N).
 * Not directly used by the `solve` function in this particular problem.
 */
void sieve() {
  ll i, j, k; // Loop counters.
  prime[0] = prime[1] = false; // 0 and 1 are not prime.
  /**
   * Block Logic: Iterates from 2 up to sqrt(N) to find prime numbers.
   * Invariant: At the start of each iteration, 'i' is a potential prime number.
   */
  for (i = 2; i * i <= N; i++) {
    if (prime[i]) { // If 'i' is prime.
      /**
       * Block Logic: Marks all multiples of 'i' as non-prime.
       * Invariant: 'j' iterates through multiples of 'i', starting from i*i.
       */
      for (j = i * i; j <= N; j += i) {
        prime[j] = false;
      }
    }
  }
}

// Exponentiation
/**
 * @brief Calculates a raised to the power of b (a^b).
 * This version handles potentially large results without modular arithmetic.
 * Time Complexity: O(log b).
 * Space Complexity: O(1).
 * Not directly used by the `solve` function.
 */
ll power(ll a, ll b) {
  ll r = 1; // Result accumulator.
  /**
   * Block Logic: Binary exponentiation (exponentiation by squaring).
   * Invariant: 'r' stores (initial_a)^(initial_b - b) * (current_a)^(b if b is odd else 0).
   */
  while (b) {
    if (b % 2 == 1) // If 'b' is odd.
      r = r * a;    // Multiply result by 'a'.
    b /= 2;         // Divide 'b' by 2.
    a *= a;         // Square 'a'.
  }
  return r;
}

/**
 * @brief Calculates (a raised to the power of b) modulo m (a^b % m).
 * This version is for modular exponentiation, useful in number theory problems.
 * Time Complexity: O(log b).
 * Space Complexity: O(1).
 * Not directly used by the `solve` function.
 */
ll power(ll a, ll b, ll m) {
  a = a % m; // Ensures 'a' is within modulo range.
  ll r = 1;  // Result accumulator.
  /**
   * Block Logic: Binary modular exponentiation.
   * Invariant: 'r' stores (initial_a)^(initial_b - b) * (current_a)^(b if b is odd else 0) % m.
   */
  while (b) {
    if (b % 2 == 1) // If 'b' is odd.
      r = (r * a) % m; // Multiply result by 'a' modulo m.
    b /= 2;            // Divide 'b' by 2.
    a = (a * a) % m;   // Square 'a' modulo m.
  }
  return r;
}

// Prime Factors
/**
 * @brief Finds all factors (divisors) of a given number n.
 *
 * This function iterates up to sqrt(n) to find pairs of factors.
 * Time Complexity: O(sqrt(n)).
 * Space Complexity: O(number of factors).
 * Not directly used by the `solve` function.
 */
vector<ll> factors(ll n) {
  vector<ll> r; // Vector to store factors.
  ll i, j;      // Loop counters.
  /**
   * Block Logic: Iterates from 1 up to sqrt(n) to find factors.
   * Invariant: 'i' is a potential factor.
   */
  for (i = 1; i * i <= n; i++) {
    if (n % i == 0) { // If 'i' is a factor.
      r.push_back(i); // Add 'i' to the list.
      if (n / i != i) // If n/i is a distinct factor.
        r.push_back(n / i); // Add n/i to the list.
    }
  }
  return r;
}

// GCD (Greatest Common Divisor)
/**
 * @brief Calculates the Greatest Common Divisor (GCD) of two numbers using Euclidean algorithm.
 * Time Complexity: O(log(min(a, b))).
 * Space Complexity: O(log(min(a, b))) due to recursion stack, or O(1) iterative.
 * Not directly used by the `solve` function.
 */
ll gcd(ll a, ll b) {
  // Ensures 'a' is always greater than or equal to 'b' for consistent modulo operation.
  if (b > a) {
    ll t = a;
    a = b;
    b = t;
  }
  if (b == 0) // Base case: GCD(a, 0) = a.
    return a;
  return gcd(b, a % b); // Recursive call using the Euclidean algorithm.
}

/**
 * @brief Calculates the Least Common Multiple (LCM) of two numbers.
 * Uses the formula: LCM(a, b) = (a * b) / GCD(a, b).
 * Time Complexity: O(log(min(a, b))) due to GCD.
 * Space Complexity: O(log(min(a, b))) due to GCD.
 * Not directly used by the `solve` function.
 */
ll lcm(ll a, ll b) { return (a * b) / gcd(a, b); }

// Graphs
// Declares a boolean vector 'bvisited' to keep track of visited nodes in BFS.
vector<bool> bvisited(N, false);

/**
 * @brief Performs a Breadth-First Search (BFS) on a graph.
 *
 * This function traverses a graph starting from a source node 's', exploring
 * all neighbor nodes at the current depth before moving to the next depth level.
 * @param V (ll): The number of vertices in the graph (not directly used in this implementation).
 * @param G (vector<vector<ll>>): Adjacency list representation of the graph.
 * @param s (ll): The starting node for the BFS.
 * @return vector<ll>: A vector containing the nodes in BFS traversal order.
 * Time Complexity: O(V + E), where V is vertices and E is edges.
 * Space Complexity: O(V) for visited array and queue.
 * Not directly used by the `solve` function.
 */
vector<ll> bfs(ll V, vector<vector<ll>> G, ll s) {
  vector<ll> order; // Stores the BFS traversal order.
  queue<ll> q;      // Queue for BFS traversal.
  bvisited[s] = true; // Mark source node as visited.
  q.push(s);          // Add source node to queue.
  /**
   * Block Logic: Continues as long as there are nodes to visit in the queue.
   * Invariant: 'q' contains all nodes to be visited at the current or next depth level.
   */
  while (!q.empty()) {
    s = q.front();    // Get the front node.
    order.push_back(s); // Add it to the order.
    q.pop();          // Remove from queue.
    /**
     * Block Logic: Explores all neighbors of the current node.
     * Invariant: 'x' iterates through neighbors of node 's'.
     */
    for (auto x : G[s]) {
      if (!bvisited[x]) { // If neighbor not visited.
        bvisited[x] = true; // Mark as visited.
        q.push(x);          // Add to queue.
      }
    }
  }
  return order;
}

// Declares a boolean vector 'dvisited' to keep track of visited nodes in DFS.
vector<bool> dvisited(N, false);

/**
 * @brief Performs a Depth-First Search (DFS) on a graph.
 *
 * This function traverses a graph starting from a source node 's', exploring
 * as far as possible along each branch before backtracking.
 * @param V (ll): The number of vertices in the graph (not directly used in this implementation).
 * @param G (vector<vector<ll>>): Adjacency list representation of the graph.
 * @param s (ll): The starting node for the DFS.
 * @return vector<ll>: A vector containing the nodes in DFS traversal order.
 * Time Complexity: O(V + E), where V is vertices and E is edges.
 * Space Complexity: O(V) for visited array and stack.
 * Not directly used by the `solve` function.
 */
vector<ll> dfs(ll V, vector<vector<ll>> G, ll s) {
  vector<ll> order; // Stores the DFS traversal order.
  stack<ll> q;      // Stack for DFS traversal.
  q.push(s);        // Add source node to stack.
  /**
   * Block Logic: Continues as long as there are nodes to visit in the stack.
   * Invariant: 'q' contains nodes to be visited.
   */
  while (!q.empty()) {
    ll v = q.top(); // Get the top node.
    q.pop();        // Remove from stack.
    if (!dvisited[v]) { // If not visited.
      order.push_back(v); // Add to order.
      dvisited[v] = true; // Mark as visited.
    }
    /**
     * Block Logic: Explores all neighbors of the current node.
     * Invariant: 'i' iterates through neighbors of node 'v'.
     */
    for (ll i = 0; i < G[v].size(); i++) {
      if (!dvisited[G[v][i]]) { // If neighbor not visited.
        q.push(G[v][i]);        // Add to stack.
      }
    }
  }
  return order;
}

// Main Solution
/**
 * @brief Solves the "Nearly Lucky Number" problem.
 * Reads a number and counts the occurrences of lucky digits (4 and 7).
 * Then, it checks if this count itself is composed only of lucky digits.
 * If both conditions are met, the number is "nearly lucky".
 * Time Complexity: O(log10(N)) for the input number, O(log10(count)) for the count.
 * Space Complexity: O(1).
 */
void solve() {
  ll n, i, j, k, l = 0; // 'n' for input number, 'l' for count of lucky digits in 'n'.
  cin >> n;            // Reads the input number.
  /**
   * Block Logic: Counts lucky digits (4 and 7) in the input number 'n'.
   * Invariant: 'n' is progressively reduced by removing its last digit. 'l' accumulates lucky digit count.
   */
  while (n) {
    if (n % 10 == 4 || n % 10 == 7) // Checks if the last digit is 4 or 7.
      l++;                          // Increments lucky digit count.
    n /= 10;                        // Removes the last digit.
  }
  k = 0, j = 0; // 'k' for lucky digit count in 'l', 'j' for total digits in 'l'.
  ll u = l;     // Stores the original count of lucky digits for final check.
  /**
   * Block Logic: Counts lucky digits (4 and 7) in the 'l' (the count of lucky digits from original number).
   * Also counts total digits in 'l'.
   * Invariant: 'l' is progressively reduced. 'j' and 'k' accumulate counts.
   */
  while (l) {
    j++;                            // Increments total digit count in 'l'.
    if (l % 10 == 4 || l % 10 == 7) // Checks if the last digit of 'l' is 4 or 7.
      k++;                          // Increments lucky digit count in 'l'.
    l /= 10;                        // Removes the last digit of 'l'.
  }
  // Conditional Logic: Checks if 'l' (original lucky digit count) itself consists only of lucky digits,
  // and if the original lucky digit count was greater than zero.
  if (j == k && u > 0) // If total digits in 'l' equals lucky digits in 'l' AND original lucky count > 0.
    cout << "YES";     // It's a "nearly lucky number".
  else
    cout << "NO";      // Not a "nearly lucky number".
}

int main() {
  /**
   * Functional Utility: Optimizes C++ standard stream I/O operations.
   * `ios_base::sync_with_stdio(false)` unties C++ streams from C standard streams.
   * `cin.tie(NULL)` prevents `cin` from flushing `cout` before each input operation.
   * `cout.tie(NULL)` prevents `cout` from flushing before each output operation.
   * These are common competitive programming optimizations for performance-critical tasks.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  
  ll t = 1; // 't' stores the number of test cases, initialized to 1 as common for single-testcase problems.
  /**
   * Block Logic: Loop for handling multiple test cases.
   * Invariant: 't' is decremented after each call to `solve()`.
   * For this specific problem, it runs `solve()` once.
   */
  while (t--) {
    solve(); // Calls the core solution function.
  }
  return 0; // Indicates successful program execution.
}
