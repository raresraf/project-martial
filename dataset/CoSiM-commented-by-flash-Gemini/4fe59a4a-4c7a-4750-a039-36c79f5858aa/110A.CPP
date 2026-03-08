/**
 * @file 110A.CPP
 * @brief Determines if an input number is a "Nearly Lucky Number".
 *        A number is nearly lucky if the count of its 'lucky digits' (4s and 7s) is itself a lucky number (4 or 7).
 *
 * Problem Name: Nearly Lucky Number (Codeforces 110A)
 * Algorithm: Counts occurrences of lucky digits in the input number string, then checks if this count is 4 or 7.
 * Time Complexity: O(L) where L is the number of digits in the input number (length of the string).
 * Space Complexity: O(L) for storing the input number as a string.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, e.g., <iostream>, <string>.

// Macros for common loop constructs and vector operations in competitive programming.
#define FOR(i, a, n) for (int i = (int)(a); i < (int)(n); ++i) // Functional Utility: Loop from 'a' to 'n'-1.
#define REP(i, n) FOR(i, 0, n)               // Functional Utility: Loop from 0 to 'n'-1.
#define all(a) a.begin(), a.end()           // Functional Utility: Returns iterators to the beginning and end of a container.
#define pb push_back                        // Functional Utility: Shorthand for vector::push_back.

// Typedefs for common competitive programming data types.
typedef unsigned long long llu; // Functional Utility: Shorthand for unsigned long long.
typedef long long ll;           // Functional Utility: Shorthand for long long.
typedef long double ld;         // Functional Utility: Shorthand for long double.

using namespace std;

/**
 * @brief Main function to check if a number is a Nearly Lucky Number.
 * @pre Input consists of a single number as a string.
 * @post Prints "YES" if the number is nearly lucky, otherwise "NO".
 */
int main() {
  string line; // Variable to store the input number as a string.
  // Functional Utility: Reads the entire line from standard input into 'line'.
  getline(cin, line);

  int cnt = 0; // Counter for lucky digits ('4' and '7').

  /**
   * Block Logic: Iterates through each character (digit) of the input string to count lucky digits.
   * Invariant: `cnt` increments each time a '4' or '7' is encountered.
   */
  REP(i, line.size()) { // Loop through the characters of the string.
    /**
     * Block Logic: Checks if the current digit is a lucky digit.
     * Pre-condition: `line[i]` is the current character being examined.
     * Invariant: If `line[i]` is '4' or '7', it's a lucky digit.
     */
    if (line[i] == '4' || line[i] == '7')
      cnt++; // Increment lucky digit counter.
  }

  // Functional Utility: Prints "YES" or "NO" based on whether 'cnt' is 4 or 7.
  printf("%s", cnt == 4 || cnt == 7 ? "YES\n" : "NO\n");
  // Indicates successful program execution.
  return 0;
}
