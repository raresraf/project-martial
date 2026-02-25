/**
 * @file 546A.CPP
 * @brief Solves a problem that involves counting elements in a sorted list
 *        that satisfy a given condition. Specifically, it reads 'n' pairs
 *        (presumably representing intervals or start/end points), sorts them,
 *        and then finds how many of these intervals have their second element
 *        (end point) greater than or equal to a given value 'k'.
 *        This is a common pattern in competitive programming for range queries
 *        or interval management.
 * Algorithm: Sorting and Linear Scan
 * Time Complexity: O(N log N) due to the sorting step, where N is the number of pairs.
 *                  The subsequent linear scan is O(N).
 * Space Complexity: O(N) to store the pairs.
 */

/* winners never quit and quitters never win.
      #swap                                 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.
using namespace std;

// Type definitions for convenience and readability.
typedef long long int ll;   ///< @brief Defines 'll' as a shorthand for long long int.
typedef long double ld;     ///< @brief Defines 'ld' as a shorthand for long double.

// Common constants and macros.
#define mod 1000000007                        ///< @brief A common modulo value used in problems involving large numbers.
#define inf 1000000000000000000               ///< @brief A large value representing infinity, often used for initial minimum/maximum.
#define rk 256                                ///< @brief Unused in this specific code, but often used as a constant or for hashing.
#define pb push_back                          ///< @brief Shorthand for vector::push_back.
#define mp make_pair                          ///< @brief Shorthand for std::make_pair, used for creating pairs.
#define vc vector                             ///< @brief Shorthand for std::vector.
#define fs first                              ///< @brief Shorthand for accessing the first element of a pair.
#define sec second                            ///< @brief Shorthand for accessing the second element of a pair.
#define pq priority_queue                     ///< @brief Shorthand for std::priority_queue.
#define lb lower_bound                        ///< @brief Shorthand for std::lower_bound.
#define ub upper_bound                        ///< @brief Shorthand for std::upper_bound.
#define pll pair<ll, ll>                      ///< @brief Defines 'pll' as a shorthand for pair<long long, long long>.
#define pls pair<ll, string>                  ///< @brief Defines 'pls' as a shorthand for pair<long long, string>.
#define psl pair<string, ll>                  ///< @brief Defines 'psl' as a shorthand for pair<string, long long>.
#define plc pair<ll, char>                    ///< @brief Defines 'plc' as a shorthand for pair<long long, char>.
#define pcl pair<char, ll>                    ///< @brief Defines 'pcl' as a shorthand for pair<char, long long>.
#define pss pair<string, string>              ///< @brief Defines 'pss' as a shorthand for pair<string, string>.
#define all(v) v.begin(), v.end()             ///< @brief Shorthand for iterating over the entire range of a container.
#define tol(s) transform(s.begin(), s.end(), s.begin(), ::tolower); ///< @brief Transforms a string to lowercase.
#define tou(s) transform(s.begin(), s.end(), s.begin(), ::toupper); ///< @brief Transforms a string to uppercase.
// Macro for handling multiple test cases.
#define T                                                                      \
  ll t = 0;                                                                    \
  cin >> t;                                                                    \
  for (ll test = 0; test < t; test++)

/**
 * @brief Main function of the program.
 *
 * This function reads a list of 'n' pairs, sorts them, and then reads a value 'k'.
 * It then counts and prints the number of pairs whose second element is
 * greater than or equal to 'k'.
 *
 * @return 0 if the program executes successfully.
 */
int main() {
  // Functional Utility: Disables synchronization between C-style I/O and C++ streams.
  // This, along with `cin.tie(0)`, significantly speeds up I/O operations,
  // which is crucial for competitive programming.
  ios_base::sync_with_stdio(false);
  // Functional Utility: Unties cin from cout.
  // This prevents `cout` from flushing before each `cin` operation, further
  // improving I/O performance.
  cin.tie(0);
  cout.tie(0);
  // isprime(); // Commented out: presumably a helper function for prime number checks.
  // coeff();   // Commented out: presumably a helper function for coefficient calculations.
  /*freopen("input.txt", "r", stdin); // Commented out: for redirecting input from a file.
freopen("output.txt", "w", stdout);*/ // Commented out: for redirecting output to a file.
  
  ll n; // Declare a variable to store the number of pairs.
  // Read the number of pairs.
  cin >> n;
  
  vc<pll> v(n); // Declare a vector of pairs of long longs of size 'n'.
  
  /**
   * Block Logic: Reads 'n' pairs of long long integers into the vector 'v'.
   * Invariant: After each iteration, one pair (fs, sec) is read and stored in 'v'.
   */
  for (ll a = 0; a < n; a++)
    cin >> v[a].fs >> v[a].sec; // Read the first and second elements of each pair.
  
  // Algorithm Step: Sort the vector of pairs.
  // By default, `std::sort` for pairs sorts primarily by the first element,
  // and then by the second element if the first elements are equal.
  sort(all(v)); // Sorts the vector 'v' using the defined 'all' macro.
  
  ll k; // Declare a variable to store the target value 'k'.
  // Read the target value 'k'.
  cin >> k;
  
  /**
   * Block Logic: Iterates through the sorted vector to find the first pair
   *              whose second element is greater than or equal to 'k'.
   * Pre-condition: 'v' is sorted.
   * Invariant: 'a' iterates through the indices of 'v'.
   * Post-condition: The loop breaks upon finding the first such pair,
   *                 and the count of remaining pairs is printed.
   */
  for (ll a = 0; a < n; a++) {
    // Check if the second element of the current pair is greater than or equal to 'k'.
    if (v[a].sec >= k) {
      // If the condition is met, all subsequent pairs (due to sorting) will also satisfy it.
      // The number of such pairs is 'n - a'.
      cout << n - a << endl; // Print the count.
      break; // Exit the loop as the desired count is found.
    }
  }
  return 0; // Indicate successful program execution.
}
