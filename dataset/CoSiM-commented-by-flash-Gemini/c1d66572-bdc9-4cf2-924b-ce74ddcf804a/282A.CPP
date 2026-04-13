/**
 * @file 282A.CPP
 * @brief Solution to a competitive programming problem: "A. Bit++".
 *
 * This program simulates the execution of a simple programming language called Bit++.
 * The language operates on a single variable 'x', initially 0, and supports two operations:
 * '++' (increment 'x' by 1) and '--' (decrement 'x' by 1). Statements are
 * 3-character strings like "++X", "X++", "--X", or "X--". The program reads
 * a sequence of these statements and outputs the final value of 'x' after
 * all statements are executed.
 *
 * Algorithm: Iterative statement processing with conditional value update.
 * Time Complexity: O(N), where N is the number of statements. Each statement
 *                  is processed in constant time due to fixed string length and direct character access.
 * Space Complexity: O(1) for storing variables, plus O(S_max) for the maximum
 *                   length of a statement string (which is constant, 3).
 */
#include <bits/stdc++.h> // Common header in competitive programming, includes most standard libraries for convenience.
#include <ext/pb_ds/assoc_container.hpp> // For policy-based data structures (e.g., ordered_set/tree).
#include <ext/pb_ds/tree_policy.hpp>     // For tree_policy in policy-based data structures.

// Global macro definition: redefines 'int' to 'long long'.
// This is a common competitive programming practice to prevent integer overflow
// without explicitly writing 'long long' everywhere.
#define int long long
#define ld long double ///< @brief Alias for long double floating-point type.
#define mod 1000000007 ///< @brief Constant for modulo operation.
#define mod1 mod       ///< @brief Another modulo constant (same as mod).
#define mod2 100000009 ///< @brief A second distinct modulo constant.

// Debugging and utility macros.
#define show(a)                                                                \
  for (int i = 0; i < a.size(); i++)                                           \
    cout << a[i & 1] << " "; ///< @brief Macro to print elements of a vector (potential bug with a[i&1]).
#define fi first       ///< @brief Alias for std::pair::first.
#define se second      ///< @brief Alias for std::pair::second.
#define vi vector<int> ///< @brief Alias for a vector of (long long) integers.
#define vs vector<string> ///< @brief Alias for a vector of strings.
#define vll vector<ll> ///< @brief Alias for a vector of long long integers (conflicts with `int` redefinition).
#define pb push_back   ///< @brief Appends an element to a std::vector.
#define pi pair<int, int> ///< @brief Alias for a pair of (long long) integers.
#define si set<int>    ///< @brief Alias for a set of (long long) integers.
#define sll set<ll>    ///< @brief Alias for a set of long long integers (conflicts with `int` redefinition).
#define pll pair<ll, ll> ///< @brief Alias for a pair of long long integers (conflicts with `int` redefinition).
// Macros for fast input using scanf.
#define sci(x) scanf("%d", &x)     ///< @brief Scans an integer. Note: `%d` might conflict with `int` being `long long`.
#define scs(x) scanf("%s", x)      ///< @brief Scans a C-style string.
#define scll(x) scanf("%I64d", &x) ///< @brief Scans a long long integer (specific to some compilers like MinGW).
#define all(v) v.begin(), v.end()  ///< @brief Returns an iterator range for a container, useful for algorithms.
#define sot(v) sort(v.begin(), v.end()) ///< @brief Sorts a container using `std::sort`.
#define maxheap priority_queue<int> ///< @brief Alias for a max-priority queue of (long long) integers.
#define mp make_pair ///< @brief Creates a std::pair.
// Macro for fast I/O operations.
#define fast_io()                                                              \
  cin.sync_with_stdio(false);                                                  \
  cout.sync_with_stdio(false);
#define PI 3.1415926535 ///< @brief Mathematical constant Pi.
#define long_zero 0ll   ///< @brief Explicit long long zero.
#define long_one 1ll    ///< @brief Explicit long long one.

// Inline function to count set bits using a GCC built-in function.
inline int sbt(int x) { return __builtin_popcount(x); }

/**
 * @brief Standard namespace usage and GNU extension for policy-based data structures.
 * 'using namespace std;' is common in competitive programming.
 * '__gnu_pbds' provides advanced data structures like ordered_set.
 */
using namespace std;
using namespace __gnu_pbds;

// Looping and debugging macros.
#define FOR(i, j, k, in) for (i = j; i < k; i += in) ///< @brief Generic for loop.
#define TRACE ///< @brief Macro to enable tracing/debugging output.
#ifdef TRACE
#define deb(...) __f(#__VA_ARGS__, __VA_ARGS__) ///< @brief Debug macro for printing variable names and values.
template <typename Arg1> void __f(const char *name, Arg1 &&arg1) {
  cout << name << " : " << arg1 << std::endl;
}
template <typename Arg1, typename... Args>
void __f(const char *names, Arg1 &&arg1, Args &&...args) {
  const char *comma = strchr(names + 1, ',');
  cout.write(names, comma - names) << " : " << arg1 << " | ";
  __f(comma + 1, args...);
}
#else
#define deb(...) ///< @brief Debug macro disabled if TRACE is not defined.
#endif
#define N 10003 ///< @brief A constant for maximum size, possibly related to problem constraints.
bool ok;        ///< @brief Global boolean variable, likely a flag (unused in main).

// Type definition for an ordered set using policy-based data structures.
typedef tree<int, null_type, less<int>, rb_tree_tag,
             tree_order_statistics_node_update>
    ost;

// Inline functions for modular arithmetic operations (addition, subtraction, multiplication).
inline int mul(int a, int b, int m = mod) { return (a * b) % m; }
inline int add(int a, int b, int m = mod) { return (a + b) % m; }
inline int sub(int a, int b, int m = mod) { return (a - b + m) % m; }

/**
 * @brief Main function: entry point of the program.
 * Processes 'N' statements for the Bit++ language, updating a sum ('x' value)
 * based on increment/decrement operations, and then prints the final sum.
 * The `int32_t` return type is a C++ standard integer type with at least 32 bits.
 */
int32_t main() {
  fast_io(); // Applies fast I/O optimizations.
  int n, i;  // Declares 'n' for the number of statements, 'i' for loop counter (both are long long due to #define).
  int sum = 0; // Initializes 'sum' (representing the variable 'x') to 0.
  string s;    // Declares a string to store each statement.
  cin >> n;    // Reads the number of statements.

  /**
   * @brief Loop through each statement in the Bit++ program.
   * Block Logic: Reads each statement. Determines if it's an increment or decrement
   * operation by checking the character at index 1 (the middle character, which is always
   * the operator in "++X", "X++", "--X", "X--"). Updates 'sum' accordingly.
   * Pre-condition: 'n' holds the total number of statements to process.
   * Invariant: 'sum' accurately reflects the value of 'x' after processing all statements
   *            up to the current iteration.
   */
  for (i = 0; i < n; i++) {
    cin >> s;                      // Reads the current statement (e.g., "++X", "X--").
    sum += s[1] == '+' ? 1 : -1; // If the middle character is '+', increment sum by 1; otherwise, decrement by 1.
                                 // This concisely handles both "++X"/"X++" and "--X"/"X--" forms.
  }
  cout << sum; // Prints the final accumulated value of 'x'.
}