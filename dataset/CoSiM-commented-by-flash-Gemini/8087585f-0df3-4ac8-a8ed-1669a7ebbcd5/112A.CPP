/**
 * @file 112A.CPP
 * @brief Solution to Codeforces problem 112A - Petya and Strings.
 *        This program compares two input strings lexicographically, ignoring case differences.
 *        It outputs -1 if the first string is lexicographically smaller, 1 if it's larger,
 *        and 0 if they are equal. This implementation performs in-place conversion to lowercase
 *        and then uses standard string comparison.
 *
 * Algorithm: In-place Case Normalization followed by Lexicographical Comparison
 *   1. Reads two input strings `a` and `b`.
 *   2. Iterates through both strings simultaneously, converting each character to its
 *      lowercase equivalent directly within the original string.
 *      (e.g., if character's ASCII value is less than 92 (ASCII for '\\[') it's uppercase,
 *      so add 32 to convert to lowercase).
 *   3. After both strings are converted to lowercase, their standard lexicographical
 *      comparison is performed.
 *   4. Prints -1, 0, or 1 based on the comparison result.
 *
 * Time Complexity: O(L), where L is the length of the input strings (assuming equal length).
 *                  This involves a single linear pass to convert characters to lowercase
 *                  and another linear pass for the string comparison.
 * Space Complexity: O(L) for storing the two input strings `a` and `b`. O(1) for auxiliary
 *                   variables and the numerous unused macros/typedefs.
 */

#include <bits/stdc++.h> // Include every standard library - Common practice in competitive programming.
using namespace std;

// Typedefs for common data types to shorten code. Not directly used in this specific main function.
typedef long long LL;
typedef pair<int, int> pii;
typedef pair<LL, LL> pll;
typedef pair<string, string> pss;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<pii> vii;
typedef vector<LL> vl;
typedef vector<vl> vvl;

// Constants often used in competitive programming. Not directly used in this specific main function.
double EPS = 1e-9;      // Epsilon for floating-point comparisons.
int INF = 1000000005;   // A large integer value representing infinity.
long long INFF = 1000000000000000005LL; // A large long long value representing infinity.
double PI = acos(-1);   // Value of Pi.
int dirx[8] = {-1, 0, 0, 1, -1, -1, 1, 1}; // Directions for 8-way movement (e.g., in grids).
int diry[8] = {0, 1, -1, 0, -1, 1, -1, 1}; // Directions for 8-way movement (e.g., in grids).

// Macros for debugging and common operations. Not all are used in this main function.
#define DEBUG fprintf(stderr, "====TESTING====\n") // Debugging print.
#define VALUE(x) cerr << "The value of " << #x << " is " << x << endl // Print variable name and value.
#define debug(...) fprintf(stderr, __VA_ARGS__) // Variadic debug print.
#define FOR(a, b, c) for (int(a) = (b); (a) < (c); ++(a)) // Standard for loop.
#define FORN(a, b, c) for (int(a) = (b); (a) <= (c); ++(a)) // For loop including upper bound.
#define FORD(a, b, c) for (int(a) = (b); (a) >= (c); --(a)) // Reverse for loop.
#define FORSQ(a, b, c) for (int(a) = (b); (a) * (a) <= (c); ++(a)) // For loop for square root optimization.
#define FORC(a, b, c) for (char(a) = (b); (a) <= (c); ++(a)) // For loop for characters.
#define FOREACH(a, b) for (auto &(a) : (b)) // Range-based for loop.
#define REP(i, n) FOR(i, 0, n) // For loop from 0 to n-1.
#define REPN(i, n) FORN(i, 1, n) // For loop from 1 to n.
#define MAX(a, b) a = max(a, b) // Macro to find max and assign.
#define MIN(a, b) a = min(a, b) // Macro to find min and assign.
#define SQR(x) ((LL)(x) * (x)) // Square of a number.
#define RESET(a, b) memset(a, b, sizeof(a)) // Memory set.
#define fi first // Pair first element.
#define se second // Pair second element.
#define mp make_pair // Make pair.
#define pb push_back // Push back to vector.
#define ALL(v) v.begin(), v.end() // Iterator to beginning and end of container.
#define ALLA(arr, sz) arr, arr + sz // Pointers to beginning and end of array.
#define SIZE(v) (int)v.size() // Get size of container.
#define SORT(v) sort(ALL(v)) // Sort container.
#define REVERSE(v) reverse(ALL(v)) // Reverse container.
#define SORTA(arr, sz) sort(ALLA(arr, sz)) // Sort array.
#define REVERSEA(arr, sz) reverse(ALLA(arr, sz)) // Reverse array.
#define PERMUTE next_permutation // Generate next permutation.
#define TC(t) while (t--) // Loop for test cases.
// Macro for fast input/output operations.
#define FAST_INP                                                               \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL)

int main() {
  /**
   * @brief Main function that solves the "Petya and Strings" problem.
   *
   * Reads two strings, converts them to lowercase in-place, and then compares them
   * lexicographically, printing the result (-1, 0, or 1).
   *
   * @return int Returns 0 upon successful execution.
   */
  FAST_INP; // Optimizes C++ standard streams for faster input/output.
  string a, b; // Input strings to be compared.
  cin >> a >> b; // Reads the two strings from input.

  /**
   * Block Logic: Iterates through both strings to convert all uppercase characters to lowercase in-place.
   * Invariant: After each iteration, characters `a[i]` and `b[i]` are converted to lowercase if they were uppercase.
   * Pre-condition: `a` and `b` are valid strings of assumed equal length.
   */
  FOR(i, 0, a.length()) { // Macro for a for-loop from 0 to `a.length() - 1`.
    // Conditional Logic: Checks if `a[i]` is an uppercase letter (ASCII value < 92).
    if (a[i] < 92) // ASCII value for 'Z' is 90. Characters 'A'-'Z' are 65-90.
      a[i] += 32;  // Inline: Adds 32 to convert uppercase to lowercase (e.g., 'A'+32='a').
    // Conditional Logic: Checks if `b[i]` is an uppercase letter.
    if (b[i] < 92)
      b[i] += 32;  // Inline: Adds 32 to convert uppercase to lowercase.
  }

  /**
   * Conditional Logic: Performs lexicographical comparison of the now-lowercase strings.
   * Prints -1 if `a` < `b`, 1 if `a` > `b`, and 0 if `a` == `b`.
   */
  if (a < b)
    cout << -1; // `a` is lexicographically smaller than `b`.
  else if (a > b)
    cout << 1;  // `a` is lexicographically greater than `b`.
  else if (a == b)
    cout << 0;  // `a` is lexicographically equal to `b`.
  return 0; // Standard return for main function.
}