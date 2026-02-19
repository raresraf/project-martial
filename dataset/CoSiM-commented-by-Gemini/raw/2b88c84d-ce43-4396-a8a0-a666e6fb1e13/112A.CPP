/**
 * @file 112A.CPP
 * @brief Solution to a programming problem requiring case-insensitive lexicographical comparison of two strings.
 * @details This program reads two strings, converts them to lowercase, and then compares them to determine their lexicographical order.
 *
 * Algorithm:
 * 1. Read two strings `a` and `b`.
 * 2. Create two new character arrays, `x` and `y`, to store the lowercase versions of `a` and `b`.
 * 3. Iterate through the input strings, convert each character to lowercase, and store it in the new arrays.
 * 4. First, it performs a check for equality. If the lowercase strings are identical, it prints 0 and exits.
 * 5. If they are not identical, it uses `std::lexicographical_compare` to determine the order.
 *    - If string `a` comes before string `b`, it prints -1.
 *    - If string `b` comes before string `a`, it prints 1.
 *    - If they are equivalent, it prints 0.
 *
 * @note The use of `x + 20` and `y + 20` in `lexicographical_compare` is a potential bug. It hardcodes the comparison length to 20, which is incorrect if the strings have a different length. The comparison should use the actual length of the strings.
 *
 * Time Complexity: O(N), where N is the length of the strings, due to the loop for conversion and the comparison.
 * Space Complexity: O(N), for storing the lowercase versions of the strings in new arrays.
 */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

/**
 * @brief Converts a character to its lowercase equivalent.
 * @param x The character to convert.
 * @return The lowercase version of the character.
 */
char convert_to_lower(char x) {
  x = tolower(x);
  return x;
}

int main() {
  string a, b;
  cin >> a;
  cin >> b;
  // NOTE: Using C-style arrays 'x' and 'y' of a variable length determined at runtime
  // is a non-standard C++ extension (Variable Length Array or VLA).
  char x[a.length()], y[b.length()];
  int same = 0;

  /**
   * Block Logic: Convert both strings to lowercase and check for immediate equality.
   * Pre-condition: 'a' and 'b' are the input strings.
   * Invariant: 'x' and 'y' store the lowercase versions of 'a' and 'b'. 'same' flag is set if any character differs.
   */
  for (int i = 0; i < a.length(); i++) {
    x[i] = convert_to_lower(a[i]);
    y[i] = convert_to_lower(b[i]);
    if (x[i] != y[i]) {
      same = 1; // Flag that strings are not identical.
    }
  }
  // If strings are identical after case conversion, print 0 and terminate.
  if (same == 0) {
    cout << 0;
    return 0;
  }

  /**
   * Block Logic: Perform lexicographical comparison on the lowercase strings.
   * This determines which string would come first in dictionary order.
   * Note on implementation: The code incorrectly uses a hardcoded length of 20 for the comparison,
   * which can lead to incorrect results or undefined behavior if string lengths differ from 20.
   */
  if (lexicographical_compare(x, x + 20, y, y + 20)) {
    cout << -1; // 'a' is lexicographically smaller than 'b'.
  }
  else if (lexicographical_compare(y, y + 20, x, x + 20)) {
    cout << 1; // 'b' is lexicographically smaller than 'a'.
  } else {
    // This case should ideally not be reached if the 'same == 0' check is correct,
    // unless the string length is less than 20 and they are identical up to their length.
    cout << 0;
  }
}
