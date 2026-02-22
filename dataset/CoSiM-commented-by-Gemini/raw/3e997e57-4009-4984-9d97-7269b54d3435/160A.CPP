/**
 * @file 160A.CPP
 * @brief Solution for the "Twins" problem (Codeforces 160A).
 *
 * @algorithm
 * The problem is to find the minimum number of coins to take from a set such that
 * the sum of the taken coins is strictly greater than the sum of the remaining coins.
 *
 * A greedy approach is optimal:
 * 1. Read all coin values.
 * 2. Calculate the total sum of all coins. The target sum to beat is `total_sum / 2`.
 *    If our sum is strictly greater than `total_sum / 2`, it will also be
 *    strictly greater than the sum of the remaining coins.
 * 3. Sort the coins in descending order.
 * 4. Iterate through the sorted (largest) coins, accumulating their sum and counting
 *    how many have been taken.
 * 5. Stop as soon as the accumulated sum exceeds `total_sum / 2`. The number of
 *    coins taken at this point is the minimum required.
 *
 * @complexity
 * - Time Complexity: O(N log N), dominated by the sorting operation.
 * - Space Complexity: O(N) to store the coin values in a vector.
 */

// This section contains a large number of preprocessor macros and typedefs,
// which are common in competitive programming to shorten boilerplate code but can
// reduce readability. For example, `for0` is a for loop, `pb` is push_back, etc.
#include <bits/stdc++.h> // This wont work only for g++ compiler.

#define for0(i, n) for (int i = 0; i < (int)(n); ++i)  // 0 based indexing
#define for1(i, n) for (int i = 1; i <= (int)(n); ++i) // 1 based indexing
#define forc(i, l, r)                                                          \
  for (int i = (int)(l); i <= (int)(r);                                        \
       ++i) // closed interver from l to r r inclusive
#define forr0(i, n) for (int i = (int)(n)-1; i >= 0; --i) // reverse 0 based.
#define forr1(i, n) for (int i = (int)(n); i >= 1; --i)   // reverse 1 based

// short hand for usual tokens
#define pb push_back
#define fi first
#define se second

// to be used with algorithms that processes a container Eg: find(aint(c),42)
#define aint(x) (x).begin(), (x).end()  // Forward traversal
#define raint(x) (x).rbegin(), (x).rend() // reverse traversal

// traversal function to avoid long template definition. Now with C++11 auto
// ainteviates the pain.
#define tr(c, i)                                                               \
  for (__typeof__((c)).begin() i = (c).begin(); i != (c).end(); i++)

// find if a given value is present in a container. Container version. Runs in
// log(n) for set and map
#define present(c, x) ((c).find(x) != (c).end())

// find version works for aint containers. This is present in std namespace.
#define cpresent(c, x) (find(aint(c), x) != (c).end())

// Avoiding wrap around of size()-1 where size is a unsigned int.
#define sz(a) int((a).size())

#define FAST_INP                                                               \
  ios::sync_with_stdio(false);                                                 \
  cin.tie(0);
const unsigned int M = 1000000007;

using namespace std;

// Shorthand for commonly used types
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef pair<int, int> ii;
typedef vector<ii> vii;
typedef long long int ll;
typedef vector<int> vint;
typedef vector<vint> vvint;
typedef double ld;

int main() {
  FAST_INP; // Macro for fast I/O
  // int T;
  // cin>>T;
  // while(T--){

  ll n;
  cin >> n;
  vector<ll> coins(n);
  // Read all coin values into the vector.
  for (ll i = 0; i < n; i++)
    cin >> coins[i];

  // Sort the coins in descending order to enable the greedy approach.
  sort(coins.begin(), coins.end(), greater<int>());
  // Calculate the total sum of all coins.
  auto sum = accumulate(coins.begin(), coins.end(), 0);
  // The target is to get a sum strictly greater than half of the total.
  auto partial = sum / 2;

  auto check = 0; // The sum of coins taken so far.
  auto count = 0; // The number of coins taken so far.
  /**
   * @brief Greedily take the largest coins until their sum exceeds half the total.
   */
  for (ll i = 0; i < n; i++) {
    check += coins[i];
    count++;
    if (check > partial)
      break; // Stop as soon as the condition is met.
  }
  cout << count << "\n";
  //}
  return 0;
}