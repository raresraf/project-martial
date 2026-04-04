/**
 * @file 281A.CPP
 * @brief Capitalizes the first letter of an input string if it is a lowercase letter.
 * * Algorithm: String Character Manipulation.
 * Time Complexity: $O(1)$ as it only operates on the first character of the string.
 * Space Complexity: $O(1)$ excluding input string storage.
 *
 * This program reads a single word (string) and converts its first letter to uppercase
 * if it is initially lowercase. The rest of the string remains unchanged.
 */

#include <bits/stdc++.h> // Includes various standard libraries, e.g., for string manipulation.
#include <string>        // Specifically includes the string library.
#define ll long long     // Defines a macro for long long integer type (unused in this specific problem).
using namespace std;     // Uses the standard namespace to avoid prefixing std::.

int main() {
  // Block Logic: Conditional compilation for input/output redirection.
  // This section is common in competitive programming to allow easy testing with
  // local files ("input.txt", "output.txt") instead of standard I/O during development.
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);  // Redirects standard input to "input.txt".
  freopen("output.txt", "w", stdout); // Redirects standard output to "output.txt".
#endif

  string s; // Declares a string variable `s` to store the input word.
  cin >> s; // Reads the input word from standard input (or "input.txt").

  int x; // Declares an integer `x` to store the ASCII value of the first character.
  x = int(s[0]); // Gets the ASCII value of the first character of the string.

  // Block Logic: Checks if the first character is a lowercase letter and converts it to uppercase.
  // Pre-condition: `x` holds the ASCII value of the first character.
  // Invariant: If `x` is between ASCII 'a' (97) and 'z' (122), it's a lowercase letter.
  // Post-condition: If `s[0]` was lowercase, it becomes uppercase by subtracting 32 (ASCII difference).
  if (x >= 97) { // ASCII value for 'a' is 97.
    s[0] = x - 32; // Converts lowercase to uppercase by adjusting ASCII value.
  }

  cout << s; // Prints the modified string to standard output (or "output.txt").
  return 0;  // Standard successful exit code for main function.
}