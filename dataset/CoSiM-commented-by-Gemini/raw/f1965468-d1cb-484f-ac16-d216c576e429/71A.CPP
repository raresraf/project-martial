/**
 * @file 71A.CPP
 * @brief Abbreviates words that are "way too long".
 *
 * This program implements a solution for the "Way Too Long Words" problem. It reads
 * an integer `n` indicating the number of words to process. For each word, it checks
 * its length. If the length is greater than 10, the word is abbreviated to its first
 * character, followed by the number of characters between the first and last, and
 * finally its last character. Otherwise, the word is printed as is.
 *
 * @author f1965468-d1cb-484f-ac16-d216c576e429
 *
 * @algorithm String Manipulation / Abbreviation
 * @complexity
 * Time: O(N * L), where N is the number of words and L is the maximum length of a word.
 *       The complexity is dominated by reading the strings and the integer-to-string conversion.
 * Space: O(L), to store the longest word.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;

int n;
string s;

/**
 * @brief Converts an integer to its string representation.
 * @param cur The integer to convert.
 * @return A string representing the integer.
 * @note This is a manual implementation; `std::to_string` is standard since C++11.
 */
string convert(int cur) {
  string res;
  // Handle the case where the input is 0.
  if (cur == 0) return "0";
  /**
   * @brief Loop to extract digits.
   * Extracts digits from the least significant to the most significant and prepends
   * them to the result string.
   */
  while (cur > 0) {
    res = (char)('0' + cur % 10) + res;
    cur /= 10;
  }
  return res;
}

int main() {
  // Read the number of words to process.
  cin >> n;
  /**
   * @brief Main loop to process each word.
   * Iterates `n` times, processing one word in each iteration.
   */
  for (int i = 0; i < n; ++i) {
    cin >> s;
    /**
     * @brief Condition to check if the word is "too long".
     * A word is considered too long if its length is greater than 10.
     */
    if (s.length() <= 10) {
      // If the word is not too long, print it without modification.
      cout << s << endl;
    } else {
      /**
       * @brief Abbreviation logic.
       * Creates a new string `t` consisting of the first character,
       * the length of the middle part, and the last character.
       */
      string t = s[0] + convert(s.length() - 2) + s[s.length() - 1];
      cout << t << endl;
    }
  }

  return 0;
}
