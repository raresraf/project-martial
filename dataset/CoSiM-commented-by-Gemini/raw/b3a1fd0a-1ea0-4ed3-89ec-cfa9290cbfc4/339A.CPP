/**
 * @file 339A.CPP
 * @brief Solution to Codeforces problem 339A - "Helpful Maths".
 * @details This program sorts the numbers in a string representing a sum.
 *          For example, an input "3+2+1" would be rearranged to "1+2+3".
 *
 * Algorithm:
 * 1. Read the input string which is a sum of single-digit numbers.
 * 2. The code iterates through the string, only considering characters at even
 *    indices, which correspond to the numbers.
 * 3. It uses a nested loop structure to perform an in-place sort on these digit
 *    characters. The logic is equivalent to a Bubble Sort.
 * 4. The characters representing the numbers are swapped to place them in
 *    non-decreasing order.
 * 5. The modified string with the sorted sum is printed.
 *
 * Time Complexity: O(N^2), where N is the length of the string, due to the nested sorting loops.
 * Space Complexity: O(N) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  /**
   * @brief A nested loop structure to sort the digit characters.
   * This implements a variation of Bubble Sort that only acts on elements
   * at even indices (the numbers), leaving the '+' signs untouched.
   */
  for (int i = 0; i < s.size(); i += 2) {
    for (int j = 0; j < i; j += 2) {
      // If a smaller number is found later in the string, swap it
      // with the number at the current position `j`.
      if (s[i] < s[j]) {
        swap(s[i], s[j]);
      }
    }
  }
  // Output the rearranged string.
  cout << s;
  return 0;
}
