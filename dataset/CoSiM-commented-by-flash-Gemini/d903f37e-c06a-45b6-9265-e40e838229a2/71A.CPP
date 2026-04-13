/**
 * @file 71A.CPP
 * @brief Solution to a competitive programming problem: "Way Too Long Words".
 *
 * This program implements a common competitive programming task where words longer
 * than 10 characters are abbreviated. The abbreviation consists of the first character,
 * followed by the count of characters between the first and last, and then the last character.
 * Words 10 characters or shorter remain unchanged.
 *
 * Algorithm: Conditional string abbreviation.
 * Time Complexity: O(L) per test case, where L is the length of the input string.
 *                  Overall, O(T * L_max), where T is the number of test cases and L_max
 *                  is the maximum length of a word.
 * Space Complexity: O(L_max) to store the input string.
 */
#include <bits/stdc++.h> // Include every standard library - common in competitive programming for convenience.

/**
 * @brief Standard namespace usage.
 * In competitive programming, 'using namespace std;' is common practice to avoid
 * repeatedly prefixing standard library elements with 'std::'.
 */
using namespace std;

// Type definitions for common data types, enhancing readability and brevity.
typedef long long LL;               ///< @brief Alias for long long integer.
typedef pair<int, int> pii;         ///< @brief Alias for a pair of integers.
typedef pair<LL, LL> pll;           ///< @brief Alias for a pair of long long integers.
typedef pair<string, string> pss;   ///< @brief Alias for a pair of strings.
typedef vector<int> vi;             ///< @brief Alias for a vector of integers.
typedef vector<vi> vvi;             ///< @brief Alias for a 2D vector of integers.
typedef vector<pii> vii;            ///< @brief Alias for a vector of pairs of integers.
typedef vector<LL> vl;              ///< @brief Alias for a vector of long long integers.
typedef vector<vl> vvl;             ///< @brief Alias for a 2D vector of long long integers.

// Constants for various uses.
double EPS = 1e-9;                  ///< @brief Epsilon for floating-point comparisons.
int INF = 1000000005;               ///< @brief Integer infinity.
long long INFF = 1000000000000000005LL; ///< @brief Long long integer infinity.
double PI = acos(-1);               ///< @brief Mathematical constant Pi.
// Direction arrays for 8-directional movement (e.g., in grid problems).
int dirx[8] = {-1, 0, 0, 1, -1, -1, 1, 1};
int diry[8] = {0, 1, -1, 0, -1, 1, -1, 1};

// Macros for debugging, looping, and container manipulation.
#define DEBUG fprintf(stderr, "====TESTING====\n") ///< @brief Debug macro to print a testing message.
#define VALUE(x) cerr << "The value of " << #x << " is " << x << endl ///< @brief Debug macro to print variable name and value.
#define debug(...) fprintf(stderr, __VA_ARGS__) ///< @brief Debug macro for formatted output to stderr.
// Looping macros.
#define FOR(a, b, c) for (int(a) = (b); (a) < (c); ++(a)) ///< @brief Loop from 'b' (inclusive) to 'c' (exclusive).
#define FORN(a, b, c) for (int(a) = (b); (a) <= (c); ++(a)) ///< @brief Loop from 'b' (inclusive) to 'c' (inclusive).
#define FORD(a, b, c) for (int(a) = (b); (a) >= (c); --(a)) ///< @brief Loop from 'b' (inclusive) down to 'c' (inclusive).
#define FORSQ(a, b, c) for (int(a) = (b); (a) * (a) <= (c); ++(a)) ///< @brief Loop for values whose square is less than or equal to 'c'.
#define FORC(a, b, c) for (char(a) = (b); (a) <= (c); ++(a)) ///< @brief Loop for characters from 'b' to 'c'.
#define FOREACH(a, b) for (auto &(a) : (b)) ///< @brief Range-based for loop.
#define REP(i, n) FOR(i, 0, n) ///< @brief Loop from 0 (inclusive) to 'n' (exclusive).
#define REPN(i, n) FORN(i, 1, n) ///< @brief Loop from 1 (inclusive) to 'n' (inclusive).
// Utility macros.
#define MAX(a, b) a = max(a, b) ///< @brief Macro to update 'a' with the maximum of 'a' and 'b'.
#define MIN(a, b) a = min(a, b) ///< @brief Macro to update 'a' with the minimum of 'a' and 'b'.
#define SQR(x) ((LL)(x) * (x)) ///< @brief Macro to calculate the square of 'x', cast to LL.
#define RESET(a, b) memset(a, b, sizeof(a)) ///< @brief Macro to initialize memory with a value.
#define fi first               ///< @brief Alias for std::pair::first.
#define se second              ///< @brief Alias for std::pair::second.
#define mp make_pair           ///< @brief Alias for std::make_pair.
#define pb push_back           ///< @brief Alias for std::vector::push_back.
// Container manipulation macros.
#define ALL(v) v.begin(), v.end() ///< @brief Returns iterators to the beginning and end of a container.
#define ALLA(arr, sz) arr, arr + sz ///< @brief Returns pointers to the beginning and end of a C-style array.
#define SIZE(v) (int)v.size() ///< @brief Returns the size of a container, cast to int.
#define SORT(v) sort(ALL(v)) ///< @brief Sorts a container using `std::sort`.
#define REVERSE(v) reverse(ALL(v)) ///< @brief Reverses elements in a container.
#define SORTA(arr, sz) sort(ALLA(arr, sz)) ///< @brief Sorts a C-style array.
#define REVERSEA(arr, sz) reverse(ALLA(arr, sz)) ///< @brief Reverses elements in a C-style array.
#define PERMUTE next_permutation ///< @brief Alias for `std::next_permutation`.
// Test case and fast I/O macros.
#define TC(t) while (t--) ///< @brief Macro for a test case loop.
#define FAST_INP                                                               \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL) ///< @brief Disables synchronization with C standard streams and unties cin from cout.

/**
 * @brief Main function: entry point of the program.
 * Reads the number of test cases, and for each test case, reads a word and
 * applies the abbreviation rule if its length exceeds 10 characters.
 */
int main() {
  FAST_INP; // Applies fast I/O optimizations.
  int t;    // Declares 't' to store the number of test cases.
  cin >> t; // Reads the number of test cases.

  /**
   * @brief Loop through each test case.
   * Block Logic: Each iteration processes a single word, applying the length-based
   * abbreviation rule.
   * Invariant: `t` decreases with each iteration, ensuring all test cases are processed.
   */
  TC(t) { // Uses the TC macro for the test case loop.
    string x; // Declares 'x' to store the input word.
    cin >> x; // Reads the input word.

    /**
     * @brief Conditional abbreviation logic.
     * Block Logic: Checks if the word's length is greater than 10.
     * If true, it prints the abbreviated form (first char, length-2, last char).
     * Otherwise, it prints the word as is.
     */
    if (x.length() > 10)
      // Prints the abbreviated word: first character, number of intermediate characters, last character.
      // Example: "internationalization" -> "i18n".
      cout << x[0] << x.length() - 2 << x[x.length() - 1] << "\n";
    else
      cout << x << "\n"; // Prints the original word if its length is 10 or less.
  }
}