/**
 * @file 160A.CPP
 * @brief Solution for competitive programming problem: "Twins".
 *
 * @details This program aims to find the minimum number of coins a person needs
 *          to take from a given set of coins such that the sum of their chosen
 *          coins is strictly greater than the sum of the remaining coins.
 *          Algorithm: Greedy approach. Sort all coins in descending order and
 *                     iteratively pick the largest coins until the sum of
 *                     picked coins exceeds half of the total sum.
 *          Time Complexity: O(N log N) due to sorting, where N is the number of coins.
 *          Space Complexity: O(N) to store the coins.
 *          Domain: Competitive Programming.
 */
#include <bits/stdc++.h> // This wont work only for g++ compiler.

// Macro for 0-based indexing loop.
#define for0(i, n) for (int i = 0; i < (int)(n); ++i)  // 0 based indexing
// Macro for 1-based indexing loop.
#define for1(i, n) for (int i = 1; i <= (int)(n); ++i) // 1 based indexing
// Macro for closed interval loop from l to r inclusive.
#define forc(i, l, r)                                                          \
  for (int i = (int)(l); i <= (int)(r);                                        \
       ++i) // closed interver from l to r r inclusive
// Macro for reverse 0-based indexing loop.
#define forr0(i, n) for (int i = (int)(n)-1; i >= 0; --i) // reverse 0 based.
// Macro for reverse 1-based indexing loop.
#define forr1(i, n) for (int i = (int)(n); i >= 1; --i)   // reverse 1 based

// short hand for usual tokens
#define pb push_back // Push back to a container (e.g., vector).
#define fi first     // Access the first element of a pair.
#define se second    // Access the second element of a pair.

// to be used with algorithms that processes a container Eg: find(aint(c),42)
// Macro for forward traversal of a container's iterators.
#define aint(x) (x).begin(), (x).end()  // Forward traversal
// Macro for reverse traversal of a container's iterators.
#define raint(x) (x).rbegin, (x).rend() // reverse traversal

// traversal function to avoid long template definition. Now with C++11 auto
// ainteviates the pain.
// Macro for iterating through a container.
#define tr(c, i)                                                               \
  for (__typeof__((c)).begin() i = (c).begin(); i != (c).end(); i++)

// find if a given value is present in a container. Container version. Runs in
// log(n) for set and map
// Macro to check if an element is present in a sorted container (like set/map).
#define present(c, x) ((c).find(x) != (c).end())

// find version works for aint containers. This is present in std namespace.
// Macro to check if an element is present in a generic container.
#define cpresent(c, x) (find(aint(c), x) != (c).end())

// Avoiding wrap around of size()-1 where size is a unsigned int.
// Macro to get the size of a container as an int, avoiding unsigned integer issues.
#define sz(a) int((a).size())

// Macro for fast input/output operations.
#define FAST_INP                                                               \
  ios::sync_with_stdio(false);                                                 \
  cin.tie(0);
const unsigned int M = 1000000007; // A common modulo constant in competitive programming.

using namespace std; // Use the standard namespace to avoid prefixing std::.

// Shorthand for commonly used types
typedef vector<int> vi;       // Vector of integers.
typedef vector<vi> vvi;       // Vector of vectors of integers.
typedef pair<int, int> ii;    // Pair of integers.
typedef vector<ii> vii;       // Vector of pairs of integers.
typedef long long int ll;     // Long long integer type.
typedef vector<int> vint;     // Vector of integers (duplicate of vi).
typedef vector<vint> vvint;   // Vector of vectors of integers (duplicate of vvi).
typedef double ld;            // Long double (floating-point type).

int main() {
  FAST_INP; // Apply fast input/output optimization.
  // The following lines are commented out, suggesting they might be part of a template
  // for problems with multiple test cases, but not used in this specific solution.
  // int T;
  // cin>>T;
  // while(T--){

  ll n; // Declare 'n' for the number of coins.
  cin >> n; // Read the number of coins.
  vector<ll> coins(n); // Declare a vector to store the coin values.
  
  // Block Logic: Read all coin values into the 'coins' vector.
  for (ll i = 0; i < n; i++)
    cin >> coins[i];

  // Sort the coins in descending order. This is a greedy strategy:
  // by picking the largest coins first, we ensure the minimum count to exceed the sum.
  sort(coins.begin(), coins.end(), greater<int>());
  
  // Calculate the total sum of all coins.
  auto sum = accumulate(coins.begin(), coins.end(), 0);
  // Calculate the target partial sum: slightly more than half of the total sum.
  // If a sum is strictly greater than the remaining, it means it's greater than half.
  auto partial = sum / 2;

  // 'check' accumulates the sum of coins taken, 'count' tracks the number of coins taken.
  auto check = 0, count = 0;
  // Block Logic: Greedily pick coins from the largest until their sum exceeds 'partial'.
  // Invariant: 'check' is the sum of 'count' largest coins.
  for (ll i = 0; i < n; i++) {
    check += coins[i]; // Add the current largest coin to 'check'.
    count++;           // Increment the count of coins taken.
    // Precondition: If the sum of taken coins ('check') exceeds 'partial', the condition is met.
    if (check > partial)
      break; // Exit the loop as the minimum number of coins has been found.
  }
  cout << count << "\n"; // Output the minimum number of coins.
  //} // Closing bracket for the commented-out test case loop.
  return 0; // Indicate successful program termination.
}