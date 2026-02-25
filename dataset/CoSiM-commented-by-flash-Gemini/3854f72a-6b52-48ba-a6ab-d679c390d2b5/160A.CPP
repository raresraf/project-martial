/**
 * @file 160A.CPP
 * @brief Solution for Codeforces Problem 160A - Twins.
 *
 * This program solves a problem where two twin brothers want to divide a set
 * of coins. One brother wants to take the minimum possible number of coins
 * such that the sum of his coins is strictly greater than the sum of the
 * remaining coins. The objective is to find this minimum number of coins.
 *
 * Algorithm: Greedy approach with sorting. The coins are sorted in descending
 * order, and the first brother greedily takes the largest coins one by one
 * until his sum strictly exceeds the sum of the remaining coins.
 * Time Complexity: O(N log N) due to sorting, where N is the number of coins.
 * Space Complexity: O(N) for storing the coin values.
 */

#include <bits/stdc++.h> // Includes various standard libraries (e.g., iostream, vector, algorithm).
using namespace std;     // Uses the standard namespace to avoid prefixing std::.

// Common competitive programming macros and aliases for brevity and speed.
#define gc getchar_unlocked             // Faster input (specific to GCC/Clang).
#define fo(i, n) for (i = 0; i < n; i++) // For loop from 0 to n-1.
#define Fo(i, k, n) for (i = k; k < n ? i < n : i > n; k < n ? i += 1 : i -= 1) // Flexible for loop.
#define ll long long                    // Alias for long long integer.
#define si(x) scanf("%d", &x)           // Faster integer input.
#define sl(x) scanf("%lld", &x)         // Faster long long input.
#define ss(s) scanf("%s", s)            // Faster string input.
#define pi(x) printf("%d\n", x)         // Faster integer output.
#define pl(x) printf("%lld\n", x)       // Faster long long output.
#define ps(s) printf("%s\n", s)         // Faster string output.
#define deb(x) cout << #x << "=" << x << endl // Debug print (variable name and value).
#define deb2(x, y) cout << #x << "=" << x << "," << #y << "=" << y << endl // Debug print for two variables.
#define pb push_back                    // Alias for vector.push_back.
#define mp make_pair                    // Alias for std::make_pair.
#define F first                         // Alias for std::pair::first.
#define S second                        // Alias for std::pair::second.
#define all(x) x.begin(), x.end()       // Returns iterators for entire container.
#define clr(x) memset(x, 0, sizeof(x))  // Sets memory to 0 (for arrays/structs).
#define sortall(x) sort(all(x))         // Sorts entire container.
#define tr(it, a) for (auto it = a.begin(); it != a.end(); it++) // Iterator-based for loop.
#define PI 3.1415926535897932384626     // Value of PI.

// Typedefs for common data structures.
typedef pair<int, int> pii;  // Pair of integers.
typedef pair<ll, ll> pl;    // Pair of long longs.
typedef vector<int> vi;     // Vector of integers.
typedef vector<ll> vl;      // Vector of long longs.
typedef vector<pii> vpii;   // Vector of pairs of integers.
typedef vector<pl> vpl;     // Vector of pairs of long longs.
typedef vector<vi> vvi;     // Vector of vectors of integers.
typedef vector<vl> vvl;     // Vector of vectors of long longs.

// Function declarations (utility functions, not directly used in this problem's main logic).
int mpow(int base, int exp);
void ipgraph(int m);
void dfs(int u, int par);

// Constants often used in competitive programming.
const int mod = 1000000007; // A common modulo for operations.
const int N = 3e5, M = N;   // Max array sizes or graph nodes.

//=======================

vi g[N]; // Global adjacency list for graph problems (not used in this solution).
int a[N]; // Global array, possibly for problem-specific data (not used in this solution).

int main() {
  /**
   * @brief Main function to execute the Twins problem logic.
   *
   * Reads the number of coins 'n' and then their individual values.
   * It sorts the coins in descending order. Then, it greedily picks coins
   * from the sorted list for the first twin until their accumulated sum
   * is strictly greater than the sum of the remaining coins. Finally,
   * it outputs the minimum number of coins taken by the first twin.
   */
  // Declare 'n' for the number of coins and 'i' for loop counter.
  int n, i;
  // Read the total number of coins.
  cin >> n;
  // Declare a vector 'ar' to store the coin values dynamically.
  vi ar(n);
  // Initialize 'sum' to calculate the total sum of all coins.
  ll sum = 0;
  // Block Logic: Read coin values and calculate their total sum.
  // Invariant: 'i' iterates from 0 to n-1, reading each coin and adding it to 'sum'.
  fo(i, n) {
    cin >> ar[i];     // Read current coin value.
    sum += ar[i];     // Add to total sum.
  }
  // Sort the coin values in descending order using std::greater<int>().
  // This allows the greedy approach to pick the largest coins first.
  sort(ar.begin(), ar.end(), greater<int>());
  
  // 'sum2' will store the sum of coins taken by the first twin.
  ll sum2 = 0;
  // Reset 'i' to 0 for iterating through the sorted coins.
  i = 0;
  // Block Logic: Greedily select coins for the first twin.
  // Pre-condition: 'ar' is sorted in descending order.
  // Invariant: 'sum2' accumulates the current twin's coins, 'i' counts them.
  // The loop continues as long as 'sum2' is not strictly greater than the sum of the remaining coins.
  while (i <= ar.size() && sum2 <= sum - sum2) {
    sum2 += ar[i++]; // Add the current largest coin to 'sum2' and increment 'i'.
  }
  // Output the minimum number of coins (which is 'i') that satisfy the condition.
  cout << i;

  return 0; // Indicate successful program execution.
}

/**
 * @brief Computes (base^exp) % mod.
 * @param base The base of the exponentiation.
 * @param exp The exponent.
 * @return The result of (base^exp) % mod.
 * @note This is a common utility function in competitive programming for modular exponentiation.
 */
int mpow(int base, int exp) {
  base %= mod;
  int result = 1;
  while (exp > 0) {
    if (exp & 1) // If exp is odd.
      result = ((ll)result * base) % mod;
    base = ((ll)base * base) % mod; // Square the base.
    exp >>= 1;                      // Divide exp by 2.
  }
  return result;
}

/**
 * @brief Initializes a graph's adjacency list.
 * @param n Number of nodes (unused in this specific function signature, assuming N constant).
 * @param m Number of edges.
 * @note This is a common utility function in competitive programming for graph input.
 */
void ipgraph(int n, int m) {
  int i, u, v;
  while (m--) {
    cin >> u >> v;         // Read edge endpoints.
    g[u - 1].pb(v - 1); // Add edge to adjacency list (0-indexed).
    g[v - 1].pb(u - 1); // For undirected graph, add reverse edge.
  }
}

/**
 * @brief Performs a Depth First Search (DFS) on a graph.
 * @param u The current node.
 * @param par The parent node (to avoid going back immediately).
 * @note This is a common utility function in competitive programming for graph traversal.
 */
void dfs(int u, int par) {
  for (int v : g[u]) { // Iterate through neighbors of current node 'u'.
    if (v == par)
      continue; // Skip if neighbor is the parent (avoid immediate back-tracking).
    dfs(v, u);    // Recursively call DFS for unvisited neighbors.
  }
}
