/**
 * @file 112A.CPP
 * @brief Solution to the "Petya and Strings" competitive programming problem (Codeforces 112A).
 * The problem requires comparing two strings lexicographically, ignoring case.
 * It outputs 1 if the first string is lexicographically greater, -1 if it's less, and 0 if they are equal.
 * Algorithm: Character-by-character comparison after converting to lowercase.
 * Time Complexity: O(L), where L is the length of the strings.
 * Space Complexity: O(L) for storing the input strings.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, including string and cctype.
using namespace std;

// Macro for fast I/O setup, common in competitive programming.
// Disables synchronization with C stdio and unties cin/cout.
#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
// Defines `endl` to be just a newline character for potentially faster output.
#define endl "\n"
// Redefines `int` to `long long` to prevent integer overflow in competitive programming.
#define int long long

// Constant for array/vector size, typically used in competitive programming templates.
const int N = 1e5 + 5;

// The main function where the program execution begins.
int32_t main() {
  // Applies fast I/O optimizations.
  IOS;
  string a, b; // Declares strings 'a' and 'b' for input.
  // Reads the two input strings.
  cin >> a >> b;

  /**
   * Block Logic: Iterates through the strings character by character to perform a case-insensitive lexicographical comparison.
   * Invariant: At the start of each iteration, 'i' is the current character index.
   * Pre-condition: Both strings 'a' and 'b' have the same length.
   * Post-condition: One of {1, -1, 0} is printed, and the program terminates.
   */
  for (int i = 0; i < a.length(); i++) {
    // Converts both characters to lowercase for case-insensitive comparison.
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);

    /**
     * Conditional Logic: Compares the lowercase characters.
     * If 'a[i]' is greater than 'b[i]', string 'a' is lexicographically greater.
     */
    if (a[i] > b[i]) {
      cout << 1 << endl; // Prints 1.
      return 0;          // Terminates the program.
    }
    /**
     * Conditional Logic: If 'a[i]' is less than 'b[i]', string 'a' is lexicographically smaller.
     */
    else if (a[i] < b[i]) {
      cout << -1 << endl; // Prints -1.
      return 0;           // Terminates the program.
    }
  }

  // If the loop completes, it means all characters were equal, so the strings are identical.
  cout << 0 << endl; // Prints 0.

  return 0; // Indicates successful program execution.
}
