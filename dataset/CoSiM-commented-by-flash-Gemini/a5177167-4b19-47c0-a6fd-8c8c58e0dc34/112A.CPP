/**
 * @file 112A.CPP
 * @brief Solution for Codeforces Problem 112A - "Petya and Strings".
 * This program compares two strings lexicographically, ignoring case.
 * It outputs -1 if the first string is lexicographically smaller, 1 if the
 * second string is smaller, and 0 if they are equal.
 * Algorithm: Convert both strings to lowercase, then perform a standard string comparison.
 * Time Complexity: O(L), where L is the length of the strings. The conversion to lowercase
 *                  and the comparison each take linear time.
 * Space Complexity: O(L) - to store the input strings.
 */
#include <cctype>  // Required for tolower() function.
#include <cstdio>  // Required for scanf() and printf() functions.
#include <cstring> // Required for strcmp() function.

using namespace std; // Uses the standard namespace.

int main() {
  // Declares two C-style character arrays to store the input strings.
  // Max length 100 + null terminator.
  char x[101], y[101];

  // Reads two strings from standard input.
  scanf("%s %s", &x, &y);

  /**
   * Block Logic: Converts all characters in both strings to lowercase.
   * This ensures that the subsequent lexicographical comparison is case-insensitive.
   * Invariant: After each iteration, `x[i]` and `y[i]` are converted to lowercase.
   */
  for (int i = 0; x[i] != 0; i++) { // Loop until the null terminator of string x is reached.
    x[i] = tolower(x[i]); // Converts character in string x to lowercase.
    y[i] = tolower(y[i]); // Converts character in string y to lowercase.
  }

  // Performs a lexicographical comparison of the two lowercase strings.
  // strcmp returns:
  //   - Negative value if x < y
  //   - Positive value if x > y
  //   - 0 if x == y
  printf("%d", strcmp(x, y));

  return 0; // Indicates successful program execution.
}