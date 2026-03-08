/**
 * @file 112A.CPP
 * @brief Case-insensitive lexicographical comparison of two strings.
 *
 * @details
 * This program implements a case-insensitive comparison of two input strings.
 * It reads two strings, normalizes them to lowercase, and then compares them
 * character by character. The program outputs -1 if the first string is
 * lexicographically smaller, 1 if it is larger, and 0 if they are identical.
 * This version is styled for competitive programming, using macros for performance
 * and brevity.
 *
 * @section
 * Algorithm: Single-pass, character-by-character comparison with early exit.
 * Time Complexity: O(L), where L is the length of the shorter string, due to early exit on first difference.
 * Space Complexity: O(N+M), where N and M are the lengths of the two strings.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @def IOS
 * @brief A macro to accelerate C++ I/O operations.
 * It disables synchronization with C-style I/O and unties cin/cout,
 * which is a common performance optimization in competitive programming.
 */
#define IOS                                                                    
  ios::sync_with_stdio(0);                                                     
  cin.tie(0);                                                                  
  cout.tie(0);

/**
 * @def endl
 * @brief Redefines `endl` to be a simple newline character `
`.
 * This avoids flushing the output buffer on every newline, improving performance.
 */
#define endl "
"

/**
 * @def int
 * @brief Redefines the `int` type to `long long`.
 * This is a common competitive programming practice to prevent integer overflow
 * with large inputs, although not strictly required for this specific problem.
 */
#define int long long

const int N = 1e5 + 5; // A constant, likely for array sizing in other problems, unused here.

/**
 * @brief Main function uses int32_t to avoid the `long long` redefinition.
 */
int32_t main() {
  // Apply I/O optimizations.
  IOS;
  string a, b;
  cin >> a >> b;

  /**
   * @brief Loop for character-by-character case-insensitive comparison.
   *
   * Logic: The loop iterates through the strings, converting each character to
   * lowercase before comparison. It employs an early-exit strategy, printing
   * the result and terminating as soon as the first lexicographical difference
   * is found.
   */
  for (int i = 0; i < a.length(); i++) {
    // Normalize characters to lowercase.
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);

    // Pre-condition: If characters at the current position differ, the
    // lexicographical order is determined.
    if (a[i] > b[i]) {
      cout << 1 << endl;
      return 0; // Early exit
    }

    else if (a[i] < b[i]) {
      cout << -1 << endl;
      return 0; // Early exit
    }
  }

  // Invariant: If the loop completes, it means no differences were found.
  // The strings are lexicographically identical (case-insensitively).
  cout << 0 << endl;

  return 0;
}
