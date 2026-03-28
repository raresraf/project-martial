/**
 * @file 58A.CPP
 * @brief Checks if the word "hello" is a subsequence of a given string.
 *
 * This program determines if the sequence of characters 'h', 'e', 'l', 'l', 'o'
 * can be found within a given input string, without regard to contiguity.
 *
 * @b Algorithm: The implementation uses a greedy two-pointer approach. It iterates
 *    backwards through both the input string and the target word "hello". When a
 *    matching character is found, the pointer for the target word is moved to the
 *    next character. This efficiently finds the last possible occurrence of the
 *    subsequence.
 * @b Time_Complexity: O(L), where L is the length of the input string, because it
 *    involves a single pass through the input.
 * @b Space_Complexity: O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {

  string str;
  cin >> str;
  char ch[] = {'h', 'e', 'l', 'l', 'o'};
  
  /**
   * @brief Iterates backwards through the input string `str` and the target `ch`.
   * @param i - Pointer for the input string, starting from the end.
   * @param j - Pointer for the target word "hello", also starting from the end.
   * @invariant `j` points to the index of the character we are currently looking
   *    for in the "hello" sequence.
   */
  for (int i = str.length() - 1, j = 4; i >= 0; i--) {
    // If we find the character we're looking for...
    if (str[i] == ch[j]) {
      // ...move on to the next character in the target sequence (in reverse).
      j--;
    }

    // If `j` is less than 0, we have successfully found all characters of "hello"
    // in the correct order.
    if (j < 0) {
      cout << "YES\n";
      return 0;
    }
  }

  // If the loop completes and we haven't found all characters, it's not a subsequence.
  cout << "NO\n";
  return 0;
}
