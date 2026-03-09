/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two strings.
 *
 * This program solves the "Petya and Strings" problem from Codeforces. It reads
 * two strings, converts them to lowercase by manually manipulating ASCII values,
 * and then compares them to determine their lexicographical order.
 *
 * @algorithm Manual String Conversion and Comparison
 * @complexity
 * - Time: O(L), where L is the length of the strings, due to the loop for
 *   case conversion and the subsequent comparison.
 * - Space: O(L) to store the two input strings.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  string s1, s2;
  // Reads two complete lines. For this problem, `cin >> s1 >> s2` would also work.
  getline(cin, s1);
  getline(cin, s2);

  /**
   * @brief Block Logic: Manually converts both strings to lowercase in-place.
   * Invariant: After this loop, both strings will be entirely lowercase.
   */
  for (int i = 0; i < s1.size(); i++) {
    // Pre-condition: Check if the character is an uppercase letter by its ASCII value.
    if (s1[i] >= 65 && s1[i] <= 91) {
      // Inline: Convert to lowercase by adding 32 to the ASCII value.
      s1[i] = s1[i] + 32;
    } else {
      // This else block is redundant as it assigns the character to itself.
      s1[i] = s1[i];
    }

    if (s2[i] >= 65 && s2[i] <= 91) {
      s2[i] = s2[i] + 32;
    } else {
      s2[i] = s2[i];
    }
  }

  /**
   * @brief Block Logic: Compares the modified strings and prints the result.
   * Uses the standard string::compare method and prints the result using printf.
   */
  if (s1.compare(s2) > 0) {
    printf("1\n");
  } else if (s1.compare(s2) < 0) {
    printf("-1\n");
  } else {
    printf("0\n");
  }

  return 0;
}