/**
 * @file 58A.CPP
 * @brief Solution to the "Chat room" problem (Codeforces 58A).
 *
 * @details
 * This program checks if a given string contains the word "hello" as a subsequence.
 * A subsequence is a sequence of characters that appear in the same order as in the
 * original string, but not necessarily consecutively. For example, "ahhelllloo"
 * contains "hello" as a subsequence.
 *
 * The program uses a greedy two-pointer approach. One pointer (`i`) iterates through
 * the input string, while a second pointer (`j`) tracks the current character we are
 * looking for in the target word "hello". When a character in the input string
 * matches the character at the target pointer `j`, `j` is advanced to look for the
 * next character of "hello".
 *
 * If the pointer `j` successfully traverses the entire word "hello" (i.e., 5 characters
 * are matched in order), the program concludes that "hello" is a subsequence.
 *
 * Algorithm: Greedy Subsequence Search
 * Time Complexity: O(L), where L is the length of the input string, because it
 *                  involves a single pass through the string.
 * Space Complexity: O(L) for storing the input string. Auxiliary space is O(1).
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  string str, str1 = "hello";
  cin >> str;

  int j = 0, pass = 0;

  /**
   * @brief Implements a greedy search for the "hello" subsequence.
   * @details This loop iterates through the input string `str` with pointer `i`.
   * The pointer `j` tracks the position in the target string `str1` ("hello").
   * When `str[i]` matches `str1[j]`, it means we have found the next character
   * of the subsequence, so we advance `j`.
   */
  for (int i = 0; i < str.length(); i++) {
    if (str[i] == str1[j]) {
      j++;
      pass++;

      // Optimization: If all 5 characters have been found, we can exit early.
      if (pass == 5)
        break;
    }
  }

  // If `pass` is 5, it means all characters of "hello" were found in order.
  if (pass == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}