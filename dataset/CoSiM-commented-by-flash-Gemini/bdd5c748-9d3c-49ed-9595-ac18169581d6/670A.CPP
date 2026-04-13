/**
 * @file 670A.CPP
 * @brief Solution to a competitive programming problem involving the calculation of "Minimum Excluded" (MEX) values.
 *
 * This program takes an array of non-negative integers and calculates the sum of two specific MEX values:
 * 1. `c1`: The smallest non-negative integer that appears less than twice in the input array.
 * 2. `c2`: The smallest non-negative integer that does not appear at all in the input array.
 *
 * Algorithm: Frequency counting using a hash map, followed by iteration to find MEX values.
 * Time Complexity: O(N) on average, where N is the number of elements in the input array.
 *                  Hash map operations (insertion, access) take O(1) on average. Finding MEX values
 *                  involves iterating from 0 upwards, which in the worst case could go up to N+1.
 * Space Complexity: O(N) to store frequencies in the hash map and distinct elements in hash sets.
 */
#include <bits/stdc++.h> // Common header in competitive programming, includes most standard libraries for convenience.

/**
 * @brief Standard namespace usage and common type aliases.
 * In competitive programming, 'using namespace std;' is common practice.
 * Type aliases simplify code and improve readability for common data structures.
 */
using namespace std;
using ll = long long;       ///< @brief Alias for long long integer.
using vll = vector<ll>;     ///< @brief Alias for a vector of long long integers.
using vvll = vector<vll>;   ///< @brief Alias for a 2D vector of long long integers.
using vvvll = vector<vvll>; ///< @brief Alias for a 3D vector of long long integers.

// Macros for common looping patterns and utilities.
#define REP(i, n, m) for (ll i = n; i < (ll)m; ++i) ///< @brief Loop from n to m-1.
#define IREP(i, n, m) for (ll i = n - 1; i >= m; --i) ///< @brief Loop from n-1 down to m.
#define rep(i, n) REP(i, 0, n) ///< @brief Loop from 0 to n-1.
#define irep(i, n) IREP(i, n, 0) ///< @brief Loop from n-1 down to 0.
#define all(v) v.begin(), v.end() ///< @brief Returns an iterator range for a container, useful for algorithms.
// Debugging macro to print elements of a vector.
#define vprint(v)                                                              \
  for (auto e : v) {                                                           \
    cout << e << " ";                                                          \
  };                                                                           \
  cout << endl;
// Debugging macro to print elements of a 2D vector.
#define vvprint(vv)                                                            \
  for (auto v : vv) {                                                          \
    vprint(v)                                                                  \
  };

/**
 * @brief Main function: entry point of the program.
 * Processes multiple test cases, each involving reading an array, calculating
 * frequencies, finding two specific MEX values (`c1` and `c2`), and printing their sum.
 */
int main() {
  cin.tie(0);                  // Unties cin from cout, potentially speeding up I/O.
  ios::sync_with_stdio(false); // Disables synchronization with C standard streams.
  cout << setprecision(20);    // Sets the precision for floating-point output.

  ll t;        // Declares 't' to store the number of test cases.
  cin >> t;    // Reads the number of test cases.

  /**
   * @brief Loop through each test case.
   * Block Logic: Each iteration solves one instance of the problem.
   * Invariant: 't' decreases with each iteration, ensuring all test cases are processed.
   */
  rep(i, t) {
    ll n;      // Declares 'n' for the size of the input array.
    cin >> n;  // Reads the size of the array.
    vll a(n);  // Declares a vector 'a' of size 'n' to store input elements.
    rep(j, n) cin >> a[j]; // Reads 'n' elements into vector 'a'.

    unordered_map<ll, ll> m; // Declares a hash map 'm' to store frequency of each number.
    // @brief Populates the frequency map.
    // Block Logic: Iterates through the input array 'a' and counts occurrences of each element.
    rep(j, n)++ m[a[j]];

    unordered_set<ll> b, c; // 'b' stores elements appearing at least twice, 'c' stores all distinct elements.
    /**
     * @brief Populates hash sets 'b' and 'c' based on element frequencies.
     * Block Logic: Iterates through the frequency map 'm'. If an element's count is 2 or more,
     * it's added to both 'b' and 'c'. Otherwise (count is 1), it's only added to 'c'.
     */
    for (auto p : m) {
      if (p.second >= 2) { // If the element appears two or more times.
        b.insert(p.first); // Add to set 'b'.
        c.insert(p.first); // Add to set 'c'.
      } else             // If the element appears exactly once.
        c.insert(p.first); // Add only to set 'c'.
    }

    ll c1 = 0, c2 = 0; // Initializes 'c1' and 'c2' which will store the MEX values.

    /**
     * @brief Finds 'c1': the smallest non-negative integer not present in set 'b'.
     * Block Logic: Increments 'c1' starting from 0 until a value is found that is NOT in set 'b'.
     * This 'c1' is the smallest non-negative integer that appears less than twice in the input array.
     */
    while (b.count(c1) > 0)
      ++c1;

    /**
     * @brief Finds 'c2': the smallest non-negative integer not present in set 'c'.
     * Block Logic: Increments 'c2' starting from 0 until a value is found that is NOT in set 'c'.
     * This 'c2' is the smallest non-negative integer that does not appear at all in the input array.
     */
    while (c.count(c2) > 0)
      ++c2;

    cout << c1 + c2 << endl; // Prints the sum of the two calculated MEX values.
  }
}
