/**
 * @file 282A.CPP
 * @brief Solution for Codeforces Problem 282A - Bit++.
 *
 * This problem involves a single variable `X` and a series of `N` operations.
 * Each operation is given as a string and can either increment `X` ("++X" or "X++")
 * or decrement `X` ("--X" or "X--"). The task is to calculate the final value of `X`
 * after all operations, starting with `X = 0`.
 *
 * Algorithm: Counter update based on string comparison.
 * The code initializes a counter `ans` to 0. For each operation string, it compares
 * the string with known increment operations ("X++", "++X"). If it matches, `ans` is
 * incremented; otherwise (implying a decrement operation), `ans` is decremented.
 *
 * Time Complexity: O(N * L), where N is the number of operations and L is the length
 * of the operation string (due to string comparison). In this case, L is constant (3).
 * So, effectively O(N).
 * Space Complexity: O(L) for storing the temporary operation string and predefined operation strings.
 */

#include <bits/stdc++.h> // Includes most standard C++ libraries, common in competitive programming.
#include <string.h> // Specifically for C-style string functions, though `<string>` is usually preferred in C++.
using namespace std; // Functional Utility: Brings all identifiers from the std namespace into the current scope.

/**
 * @brief Macro for optimizing C++ standard I/O streams.
 * Functional Utility: Disables synchronization between C++ standard streams and C standard streams,
 * and unties `cin` from `cout`. This significantly speeds up input/output operations, which is
 * crucial in competitive programming for performance.
 */
#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define endl "\n" // Macro for newline character, often faster than `std::endl`.
#define Max(x, y, z) max(x, max(y, z)) // Macro to find the maximum of three values.
#define Min(x, y, z) min(x, min(y, z)) // Macro to find the minimum of three values.
#define int long long // Functional Utility: Redefines `int` to `long long` globally.
                      // This is a common competitive programming trick to avoid integer overflow
                      // with larger input values, effectively making `int` 64-bit.
#define trace1(x) cerr << #x << ": " << x << endl // Debug macro to print variable name and value.
#define trace2(x, y) cerr << #x << ": " << x << " | " << #y << ": " << y << endl // Debug macro for two variables.
#define trace3(x, y, z)                                                        \
  cerr << #x << ":" << x << " | " << #y << ": " << y << " | " << #z << ": "    \
       << z << endl // Debug macro for three variables.

/**
 * @brief Main function: Solves the Bit++ problem.
 * Functional Utility: Reads a number of operations `n`, processes each operation
 * (increment or decrement of `X`), and prints the final value of `X`.
 * Algorithm: Simulates the operations by maintaining a counter.
 */
int32_t main() { // `int32_t` is used because `int` was redefined to `long long`.
  IOS; // Functional Utility: Apply I/O optimizations.
  int n; // Declares `n` for the number of operations.
  cin >> n; // Reads the number of operations.
  int ans = 0; // Initializes the answer (final value of X) to 0.

  // Predefined strings for comparison, representing the four possible operations.
  string temp; // String to store the current operation.
  string str1 = "X++";
  string str2 = "++X";
  string str3 = "X--"; // Unused in conditional logic, implies default decrement.
  string str4 = "--X"; // Unused in conditional logic, implies default decrement.

  // Block Logic: Loop `n` times to process each operation.
  // Precondition: `n` is a non-negative integer.
  // Invariant: `ans` reflects the sum of operations up to the current iteration.
  for (int i = 0; i < n; ++i) {
    cin >> temp; // Reads the current operation string.
    // Functional Utility: Check if the operation is an increment.
    // `string::compare` returns 0 if strings are equal.
    if ((temp.compare(str1) == 0) || (temp.compare(str2) == 0))
      ans++; // Increment `ans` if it's an "X++" or "++X" operation.
    else
      ans--; // Otherwise, it's a decrement operation ("X--" or "--X"), so decrement `ans`.
  }
  cout << ans;
  return 0; // Indicate successful program execution.
}
