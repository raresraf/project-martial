/**
 * @file 1221C.CPP
 * @brief Solution to a competitive programming problem involving resource allocation for team formation.
 *
 * This program calculates the maximum number of teams that can be formed given three types of resources:
 * 'c' (programmers), 'm' (mathematicians), and 'x' (extra specialists). The exact rules for team formation
 * are implicitly derived from the provided arithmetic logic, often simplifying to a greedy approach
 * that prioritizes balanced teams while utilizing 'x' resources effectively.
 *
 * Algorithm: Greedy resource allocation and balancing using arithmetic operations.
 * Time Complexity: O(1) per test case (query), as it involves a fixed number of arithmetic operations
 *                  and comparisons. Overall, O(Q) where Q is the number of queries.
 * Space Complexity: O(1), as only a few variables are used to store input and intermediate calculations.
 */
#include <bits/stdc++.h> // Common header in competitive programming, includes most standard libraries for convenience.
#include <ext/pb_ds/assoc_container.hpp> // For policy-based data structures (e.g., ordered_set/tree).
#include <ext/pb_ds/tree_policy.hpp>     // For tree_policy in policy-based data structures.

/**
 * @brief Standard namespace usage and GNU extension for policy-based data structures.
 * 'using namespace std;' is common in competitive programming.
 * '__gnu_pbds' provides advanced data structures like ordered_set.
 */
using namespace std;
using namespace __gnu_pbds;

// Type definitions for various data types, enhancing readability and brevity.
typedef long long ll;     ///< @brief Alias for long long integer.
typedef long double ld;   ///< @brief Alias for long double floating-point type.
typedef complex<ld> cd;   ///< @brief Alias for complex number with long double components.

typedef pair<int, int> pi; ///< @brief Alias for a pair of integers.
typedef pair<ll, ll> pl;   ///< @brief Alias for a pair of long long integers.
typedef pair<ld, ld> pd;   ///< @brief Alias for a pair of long doubles.

typedef vector<int> vi;   ///< @brief Alias for a vector of integers.
typedef vector<ld> vd;    ///< @brief Alias for a vector of long doubles.
typedef vector<ll> vl;    ///< @brief Alias for a vector of long long integers.
typedef vector<pi> vpi;   ///< @brief Alias for a vector of pairs of integers.
typedef vector<pl> vpl;   ///< @brief Alias for a vector of pairs of long long integers.
typedef vector<cd> vcd;   ///< @brief Alias for a vector of complex numbers.
typedef vector<vector<int>> vvi; ///< @brief Alias for a 2D vector of integers.
typedef vector<vector<ll>> vvl; ///< @brief Alias for a 2D vector of long long integers.
/**
 * @brief Policy-based data structure for an ordered set/tree.
 * Provides functionality similar to std::set but with additional features
 * like finding the k-th element and order of key.
 */
template <class T>
using Tree =
    tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;

// Macros for common looping patterns and utilities, optimizing competitive programming code.
#define FOR(i, a, b) for (int i = a; i < (b); i++) ///< @brief Loop from 'a' (inclusive) to 'b' (exclusive).
#define F0R(i, a) for (int i = 0; i < (a); i++) ///< @brief Loop from 0 (inclusive) to 'a' (exclusive).
#define FORd(i, a, b) for (int i = (b)-1; i >= a; i--) ///< @brief Loop from 'b'-1 (inclusive) down to 'a' (inclusive).
#define F0Rd(i, a) for (int i = (a)-1; i >= 0; i--) ///< @brief Loop from 'a'-1 (inclusive) down to 0 (inclusive).

#define sz(x) (int)(x).size() ///< @brief Returns the size of a container.
#define mp make_pair        ///< @brief Creates a std::pair.
#define pb push_back        ///< @brief Appends an element to a std::vector.
#define f first             ///< @brief Accesses the first element of a std::pair.
#define s second            ///< @brief Accesses the second element of a std::pair.
#define lb lower_bound      ///< @brief Alias for std::lower_bound.
#define ub upper_bound      ///< @brief Alias for std::upper_bound.
#define all(x) x.begin(), x.end() ///< @brief Returns iterators to the beginning and end of a container.

// Random number generator seeded with current time for varied results in each run.
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

const int MOD = 1000000007; ///< @brief Constant for modulo operation, frequently used in problems involving large numbers.
const ll INF = 1e18; ///< @brief Constant representing infinity for long long type.
const int MX = 100001; ///< @brief A maximum limit, likely for array or container sizes.

/**
 * @brief Main function: entry point of the program.
 * Reads multiple queries, each defining resource counts (c, m, x), and calculates
 * and prints the maximum number of teams that can be formed for each query.
 */
int main() {
  int q; // Declares 'q' for the number of queries/test cases.
  cin >> q; // Reads the number of queries.

  /**
   * @brief Loop through each query (test case).
   * Block Logic: Each iteration processes a distinct set of resource counts and computes
   * the maximum number of teams.
   * Invariant: 'q' decreases with each iteration, ensuring all queries are processed.
   */
  while (q--) {
    int c, m, x; // Declares 'c', 'm', 'x' for the counts of the three resource types.
    cin >> c >> m >> x; // Reads the resource counts for the current query.

    /**
     * @brief Handles the special case where 'x' (extra specialists) are abundant.
     * Block Logic: If 'x' is greater than or equal to both 'm' (mathematicians) and
     * 'c' (programmers), then the number of teams is limited by the minimum of 'm' and 'c'.
     * In this scenario, 'x' does not constrain team formation, and the maximum teams
     * are simply the smaller of the two primary resource types.
     * Pre-condition: 'c', 'm', 'x' are the current resource counts.
     */
    if (x >= m || x >= c) {
      cout << min(m, c) << endl; // Prints the minimum of 'm' and 'c'.
      continue;                  // Skips remaining logic for this query and proceeds to the next.
    }

    // Remaining logic for cases where 'x' is not abundant enough to cover all 'c' or 'm'.
    // Here, 'x' resources are effectively used to balance 'c' and 'm'.
    int rc = c - x, rm = m - x; // Calculates remaining 'c' and 'm' after using 'x' to balance.
                                // 'x' teams can always be formed initially, effectively consuming 'x' from 'c' and 'm'.

    // Ensures 'rc' always holds the larger count of the two remaining resources (programmers/mathematicians).
    // This simplification helps in the subsequent greedy distribution logic.
    if (rc < rm)
      swap(rc, rm);

    /**
     * @brief Core calculation for the maximum number of teams.
     * Block Logic: This formula combines the 'x' teams already formed with additional teams
     * that can be formed from the remaining 'rc' and 'rm' resources.
     *
     * The `min(rm, rc / 2 + max(0, (rm - rc / 2 + (rc & 1)) / 3))` part represents the additional teams:
     * - `rm`: The smaller of the two remaining resource types. This is a potential upper bound.
     * - `rc / 2`: Greedily forms pairs using the more abundant resource (`rc`).
     * - `(rc & 1)`: Accounts for any odd remaining `rc` after pairing.
     * - `(rm - rc / 2 + (rc & 1))`: Remaining `rm` after `rc` has been partially paired.
     * - `... / 3`: Distributes the *remaining* `rm` with any excess `rc` into groups of three.
     *
     * This complex expression essentially tries to form as many balanced teams as possible,
     * prioritizing groups of two and then groups of three with the remaining resources.
     */
    cout << x + min(rm, rc / 2 + max(0, (rm - rc / 2 + (rc & 1)) / 3)) << endl;
  }
}