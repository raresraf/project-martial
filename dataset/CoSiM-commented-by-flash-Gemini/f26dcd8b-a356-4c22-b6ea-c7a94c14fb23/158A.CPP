/*
Hey there stalker....
















Like what you see?
*/
/**
 * @file 158A.CPP
 * @brief Solution to a competitive programming problem: "Next Round".
 *
 * This program determines the number of participants who advance to the next round of a
 * competition. Participants advance if their score is greater than or equal to the
 * k-th place finisher's score, provided their score is also positive.
 *
 * Algorithm: Direct iteration and comparison.
 * Time Complexity: O(N) for processing the scores for each test case, where N is the number of participants.
 *                  Overall: O(T * N), where T is the number of test cases. In this specific solution, T=1.
 * Space Complexity: O(N) to store the scores of all participants.
 */
#include <algorithm> // Provides standard algorithms like std::sort.
#include <bits/stdc++.h> // Common header in competitive programming, includes most standard libraries for convenience.
#include <set>       // Provides std::set container.
#include <string>    // Provides std::string.
#include <vector>    // Provides std::vector container.

/**
 * @brief Standard namespace usage.
 * In competitive programming, 'using namespace std;' is common practice to avoid
 * repeatedly prefixing standard library elements with 'std::'.
 */
using namespace std;

// Macros for common operations and type definitions, frequently used in competitive programming for brevity.
#define IOS                                                                    \
  ios_base::sync_with_stdio(false); /* Disables synchronization with C standard streams. */ \
  cin.tie(NULL);                   /* Unties cin from cout, potentially speeding up I/O. */ \
  cout.tie(NULL);                  /* Unties cout from cin, potentially speeding up I/O. */
#define pb push_back   ///< @brief Macro for std::vector::push_back.
#define pii pair<ll, ll> ///< @brief Type definition for a pair of long long integers.
#define vi vector<ll>  ///< @brief Type definition for a vector of long long integers.
#define all(a) (a).begin(), (a).end() ///< @brief Macro for getting iterators to the beginning and end of a container.
#define F first        ///< @brief Macro for accessing the first element of a pair.
#define S second       ///< @brief Macro for accessing the second element of a pair.
#define sz(x) (ll) x.size() ///< @brief Macro for getting the size of a container, cast to long long.
#define rep(i, a, b) for (ll i = a; i < b; i++) ///< @brief Macro for a common for-loop pattern (from 'a' to 'b'-1).
#define repr(i, a, b) for (ll i = a; i > b; i--) ///< @brief Macro for a reverse for-loop pattern (from 'a' down to 'b'+1).
#define lbnd lower_bound ///< @brief Alias for std::lower_bound.
#define ubnd upper_bound ///< @brief Alias for std::upper_bound.
#define bs binary_search ///< @brief Alias for std::binary_search.
#define mp make_pair   ///< @brief Macro for std::make_pair.
#define ll long long int ///< @brief Type definition for long long integer.
#define lld long double  ///< @brief Type definition for long double.
#define endl "\n"      ///< @brief Macro for newline character, often used instead of std::endl for performance.
#define MOD 1000000007 ///< @brief Constant for modulo operation, frequently used in problems involving large numbers.

/**
 * @brief Main function: entry point of the program.
 * Processes the "Next Round" problem by reading participant scores, determining the k-th score,
 * and then counting how many participants meet the advancement criteria.
 */
int main() {
  IOS // Apply fast I/O optimizations.

  int TESTS = 1; // Declares an integer 'TESTS' to indicate the number of test cases.
  // cin >> TESTS; // Commented out, implying this solution is designed for a single test case or a specific problem setup.

  /**
   * @brief Loop through each test case.
   * Block Logic: Processes one or more problem instances. In this solution, it runs once.
   * Invariant: For each iteration, the problem's inputs are read and the solution is computed.
   */
  while (TESTS--) {
    ll n, k, a, count = 0; // Declares variables: n (number of participants), k (k-th place), a (k-th score), count (advancing participants).
    cin >> n >> k; // Reads the total number of participants (n) and the k-th position (k).

    vi v(n); // Declares a vector 'v' of size 'n' to store the scores of all participants.
    rep(i, 0, n) { cin >> v[i]; } // Reads 'n' scores into the vector 'v'.

    a = v[k - 1]; // Identifies the score of the participant in the k-th position (0-indexed).

    /**
     * @brief Counts participants who qualify for the next round.
     * Block Logic: Iterates through all scores and increments 'count' if a participant's score
     * is greater than or equal to the k-th score AND is positive.
     * Pre-condition: 'v' contains all participant scores, 'a' holds the k-th score.
     */
    rep(i, 0, n) {
      if (v[i] >= a && v[i] > 0) { // Checks if the current participant's score is non-zero and at least the k-th score.
        count++; // Increments the counter for advancing participants.
      }
    }
    cout << count << endl; // Prints the total number of participants who advance.
  }
  return 0; // Indicates successful program execution.
}
