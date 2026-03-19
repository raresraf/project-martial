/**
 * @file 112A.CPP
 * @brief Solution to Codeforces Problem 112A: "Petya and Strings".
 * Compares two input strings lexicographically, ignoring case.
 * Outputs a negative value (typically -1) if the first string is smaller,
 * a positive value (typically 1) if the first is larger, and 0 if they are equal.
 * Algorithm: Converts both strings to lowercase character by character, then uses
 * the standard C library function `strcmp` for comparison.
 * Time Complexity: O(L), where L is the maximum length of the input strings.
 * Space Complexity: O(L) to store the two C-style strings.
 */
#include <algorithm> // Included but not explicitly used for the core logic in this solution.
#include <cstring>   // Required for `strlen` and `strcmp`.
#include <iostream>  // Required for `cin`, `cout`.
#include <stdio.h>   // Required for `printf` (though `cout` is used for final output).
#include <string>    // Required for `string` class (though C-style char arrays are used).
using namespace std;

/**
 * @brief Converts an uppercase ASCII character to its lowercase equivalent.
 * @param in The input character.
 * @return The lowercase version of the character if it was uppercase, otherwise the original character.
 */
char easytolower(char in) {
  /**
   * Block Logic: Checks if the input character is an uppercase letter (A-Z).
   * If true, converts it to lowercase by adding the ASCII difference between 'a' and 'A'.
   */
  if (in <= 'Z' && in >= 'A') {
    return in - ('Z' - 'z'); // 'Z' - 'z' is -32, so subtracting it adds 32. Alternatively, `in + 32`.
  }
  return in; // Returns the character as is if it's not an uppercase letter.
}

/**
 * @brief Main function to solve the "Petya and Strings" problem.
 * Reads two strings, converts them to lowercase, compares them, and prints the result.
 */
int main() {
  char a[105], b[105]; // Declare two C-style character arrays to store input strings (max length 104 + null terminator).
  cin >> a >> b;      // Reads the two strings from standard input.
  
  int len = strlen(a); // Calculates the length of the first string. Assumes both strings have similar lengths.

  /**
   * Block Logic: Iterates through both strings (up to the length of 'a') and converts
   * each character to its lowercase equivalent using `easytolower`.
   * Pre-condition: `a` and `b` contain the input strings, `len` is the length of `a`.
   * Invariant: After each iteration, characters up to index `i` in both `a` and `b` are lowercase.
   */
  for (int i = 0; i < len; i++) {
    a[i] = easytolower(a[i]); // Convert character in `a` to lowercase.
    b[i] = easytolower(b[i]); // Convert character in `b` to lowercase.
  }
  
  // Compares the two lowercase strings lexicographically using `strcmp` from <cstring>.
  // `strcmp` returns 0 if equal, a negative value if `a` is smaller, a positive value if `a` is larger.
  cout << strcmp(a, b);
  
  return 0;
}
