/**
 * @file 112A.CPP
 * @brief This program performs a case-insensitive lexicographical comparison of two input strings.
 * It reads two strings, converts them to lowercase, and then compares them character by character
 * to determine their relative order, outputting -1, 0, or 1.
 * Algorithm: String case conversion using `std::transform` followed by manual character-by-character comparison.
 * Time Complexity: O(L) where L is the maximum length of the input strings, due to character iteration.
 * Space Complexity: O(L) for storing the input strings.
 */
#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience, includes many standard libraries.
#include <cctype>        // Required for std::tolower.
#include <string>        // Required for std::string.
#define ll long long     // Defines a macro 'll' as a shorthand for 'long long'.
using namespace std;
int main() {

// Conditional compilation block for local debugging.
// If not compiled in an online judge environment, it redirects standard input from "input.txt"
// and standard output to "output.txt".
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  int n; // Declared but not used in this specific implementation.
  int z = 0; // Stores the comparison result: -1 if a < b, 1 if a > b, 0 if a == b.
  string a, b; // Stores the two input strings.
  // Reads the two strings from standard input.
  cin >> a >> b;
  // Converts all characters in string 'a' to lowercase.
  transform(a.begin(), a.end(), a.begin(), ::tolower);
  // Converts all characters in string 'b' to lowercase.
  transform(b.begin(), b.end(), b.begin(), ::tolower);
  /**
   * Block Logic: Compares the lowercase strings character by character to determine their lexicographical order.
   * The loop iterates through the strings until a difference is found or the end of the shorter string is reached.
   * 'z' is updated to -1, 0, or 1 based on the comparison, and the loop breaks early if a difference is found.
   * Invariant: 'z' accurately reflects the lexicographical comparison result up to the current index.
   */
  for (int i = 0; i < a.length(); i++) {
    // If characters at the current position are equal, 'z' remains 0 (or retains previous value, then overwritten if needed later).
    if (int(a[i]) == int(b[i])) {
      z = 0; // Set to 0 if characters are equal. This will be updated if a difference is found later.
    }
    // If the character in 'a' is greater than in 'b', 'a' is lexicographically larger.
    else if (int(a[i]) > int(b[i])) {
      z = 1;   // Set result to 1.
      break;   // Terminate comparison as order is determined.
    }
    // If the character in 'a' is less than in 'b', 'a' is lexicographically smaller.
    else if (int(a[i]) < int(b[i])) {
      z = -1;  // Set result to -1.
      break;   // Terminate comparison as order is determined.
    }
  }
  // Prints the comparison result (-1, 0, or 1) to standard output.
  cout << z;
  // Standard program exit.
  return 0;
}
