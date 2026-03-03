
#include <algorithm>
#include <cstring>
#include <iostream>
using namespace std;

/**
 * @file 339A.CPP
 * @brief Solution to Codeforces problem 339A, "Helpful Maths".
 * @details The program takes a string representing a sum of single-digit numbers
 * (1, 2, or 3) and rearranges it into non-decreasing order. For example,
 * "3+2+1" becomes "1+2+3".
 *
 * Algorithm:
 * The core idea is to treat the input string as a character array and sort it.
 * Since the ASCII value for '+' (43) is less than the ASCII values for the
 * digits '1' through '9', sorting the string "3+2+1" results in "+123".
 * The program then iterates through the sorted array, printing only the digit
 * characters and inserting '+' separators between them. This effectively sorts
 * the numbers while ignoring the original plus signs.
 *
 * Time Complexity: O(N log N), where N is the length of the input string,
 * dominated by the `std::sort` operation.
 * Space Complexity: O(N) to store the input string.
 */
int main() {
  // Character array to store the input string, with a size buffer.
  char a[110];

  cin >> a;
  int n = strlen(a);

  // Sort the entire character array. For "3+2+1", this results in "+123".
  sort(a, a + n);

  /**
   * @brief Iterate through the sorted array to print the rearranged sum.
   *
   * Pre-condition: The character array `a` has been sorted lexicographically.
   *
   * Logic: The loop filters out the '+' characters by checking if the ASCII
   * value is greater than 43 (the ASCII code for '+'). It then prints the
   * digit and adds a '+' sign after it, unless it's the very last digit.
   */
  for (int i = 0; i < n; i++) {
    // ASCII value of '+' is 43. This condition skips over the '+' signs.
    if (a[i] > 43) {
      cout << a[i];
      // Add a '+' separator after each number, except for the last one.
      if (i + 1 < n)
        cout << "+";
    }
  }
}
