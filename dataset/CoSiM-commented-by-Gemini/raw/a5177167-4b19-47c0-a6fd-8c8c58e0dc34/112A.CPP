/**
 * @file 112A.CPP
 * @brief C-style C++ solution to the "Petya and Strings" problem.
 *
 * This program performs a case-insensitive lexicographical comparison of two
 * input strings. It outputs -1 if the first string is lexicographically smaller,
 * 1 if it's larger, and 0 if they are equal.
 *
 * @algorithm String Manipulation and Comparison
 * @time_complexity O(L), where L is the length of the strings. The program iterates
 * through the strings once for case conversion and `strcmp` also takes linear time.
 * @space_complexity O(L) to store the two input strings.
 */
#include <cctype>
#include <cstdio>
#include <cstring>

// using namespace std; // Included but only C-style functions are used.

int main() {
  char x[101], y[101];
  // Reads two whitespace-separated strings from input.
  // Note: Using &x and &y is unconventional for scanf with %s; x and y would suffice.
  scanf("%s %s", &x, &y);

  /**
   * @brief Converts both strings to lowercase for case-insensitive comparison.
   */
  for (int i = 0; x[i] != 0; i++) {
    x[i] = tolower(x[i]);
    y[i] = tolower(y[i]);
  }

  /**
   * @brief Compares the two lowercase strings and prints the result.
   * strcmp returns <0, 0, or >0, which maps directly to the problem's
   * required output of -1, 0, or 1 in this context.
   */
  printf("%d", strcmp(x, y));
  return 0;
}
