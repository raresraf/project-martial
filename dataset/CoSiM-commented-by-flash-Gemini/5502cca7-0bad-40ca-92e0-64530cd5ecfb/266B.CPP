/**
 * @file 266B.CPP
 * @brief Solution to Codeforces Problem 266B: "Queue at the School".
 *        The problem simulates a queue of boys ('B') and girls ('G').
 *        Every second, if a boy is immediately followed by a girl (BG),
 *        they swap positions (GB). This process repeats for 't' seconds.
 *
 * Algorithm: Iterative simulation. The simulation runs for `t` seconds. In each second,
 *            it performs a single pass through the queue. If a 'B' is followed by a 'G',
 *            they are swapped, and the loop index is incremented to skip the newly
 *            moved 'B' to prevent multiple swaps in a single second for the same boy.
 * Time Complexity: O(t * N) where `t` is the number of seconds and `N` is the length of the queue.
 * Space Complexity: O(N) to store the queue string.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Macro for optimizing C++ standard stream I/O operations.
 * Functional Utility: Unties C++ standard streams from C standard streams (`ios_base::sync_with_stdio(false)`),
 * unties `cin` from `cout` (`cin.tie(NULL)`), and unties `cout` from `cin` (`cout.tie(NULL)`),
 * resulting in faster input/output. This is a common optimization in competitive programming.
 */
#define faster ios_base::sync_with_stdio(false), cin.tie(NULL), cout.tie(NULL)
/**
 * @brief Macro for creating `std::pair` objects.
 * Functional Utility: Shorthand for `std::make_pair`.
 */
#define mp make_pair
/**
 * @brief Constant for a modulo value, commonly used in problems requiring large number arithmetic.
 */
#define mod 1000000007
/**
 * @brief Constant for another common modulo value, particularly in number theory problems.
 */
#define qmod 998244353
/**
 * @brief Macro for a newline character.
 * Functional Utility: Provides a portable and often faster alternative to `std::endl`.
 */
#define endl "\n"
/**
 * @brief Macro for pushing elements to the back of a container (e.g., `std::vector`).
 * Functional Utility: Shorthand for `std::vector::push_back`.
 */
#define pb push_back
/**
 * @brief Macro for accessing the first element of a pair or map entry.
 * Functional Utility: Shorthand for `std::pair::first`.
 */
#define ff first
/**
 * @brief Macro for accessing the second element of a pair or map entry.
 * Functional Utility: Shorthand for `std::pair::second`.
 */
#define ss second

/**
 * @brief Type definition for long long integer.
 * Functional Utility: Provides a shorter alias for `long long`.
 */
typedef long long ll;
/**
 * @brief Type definition for a pair of integers.
 * Functional Utility: Provides a shorter alias for `std::pair<int, int>`.
 */
typedef pair<int, int> pii;
/**
 * @brief Type definition for a vector of integers.
 * Functional Utility: Provides a shorter alias for `std::vector<int>`.
 */
typedef vector<int> vi;

/**
 * @brief Global constant for a modulo value (10^9 + 7).
 */
const int MOD = 1e9 + 7;
/**
 * @brief Global constant representing a large integer value, often used for infinity.
 */
const int INF = 1e9 + 5;
/**
 * @brief Global constant representing the maximum value for a long long integer.
 */
const ll LINF = LLONG_MAX;
int main() {
  /**
   * Functional Utility: Declaration of variables for input and queue simulation.
   * `n`: Stores the initial number of people in the queue (length of the string `s`).
   * `t`: Stores the number of seconds (time steps) for the simulation.
   * `s`: Stores the string representing the queue (sequence of 'B' and 'G').
   */
  int n;
  cin >> n; // Input: Read the number of people in the queue.
  int t;
  cin >> t; // Input: Read the number of seconds for the simulation.
  string s;
  cin >> s; // Input: Read the initial queue configuration.

  /**
   * Block Logic: Simulates the queue changes over `t` seconds.
   * Pre-condition: `t` is the number of simulation steps, `s` is the current queue state.
   * Invariant: In each iteration, one second of queue changes are applied to `s`.
   */
  while (t--) {
    /**
     * Block Logic: Performs a single pass through the queue to apply swaps for one second.
     * Iterates from the beginning of the queue up to the second to last element.
     * Pre-condition: `s` represents the queue state at the beginning of the second.
     * Invariant: `i` is the current index; `s[i]` and `s[i+1]` are checked for swaps.
     */
    for (int i = 0; i < s.size() - 1; i++) {
      /**
       * Conditional Logic: Checks for an adjacent 'B' and 'G' pair (`BG`).
       * If found, they swap positions (`GB`), and the outer loop index `i` is incremented.
       * The `i++` inside the if-block is crucial to skip the newly moved 'B' (which is now at `i+1`)
       * to ensure that each boy moves at most one position per second, as per problem rules,
       * and to prevent the same 'B' from being considered for another swap within the same second's pass.
       */
      if (s[i] == 'B' && s[i + 1] == 'G') {
        s[i] = 'G';
        s[i + 1] = 'B';
        i++; // Increment `i` to skip the girl who just moved, preventing double-swaps.
      }
    }
  }
  // Output: Print the final state of the queue after `t` seconds.
  cout << s;
  return 0;
}