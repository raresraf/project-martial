/**
 * @file 112A.CPP
 * @brief Solution for a competitive programming problem, likely Codeforces 236A - Boy or Girl.
 * @note The filename "112A.CPP" suggests "Petya and Strings", but the code logic
 *       solves the "Boy or Girl" problem, which involves counting distinct characters.
 *
 * @brief This program determines whether to "CHAT WITH HER!" or "IGNORE HIM!"
 * based on the number of distinct characters in a given user name. If the number
 * of distinct characters is even, the output is "CHAT WITH HER!"; otherwise, it's
 * "IGNORE HIM!".
 *
 * Algorithm:
 * 1. Read the input string (user name).
 * 2. Sort the string lexicographically. This places all identical characters
 *    adjacent to each other.
 * 3. Iterate through the sorted string and count the number of unique characters.
 *    This is done by comparing each character to the next one. If they are different,
 *    it signifies a new unique character.
 * 4. Check if the final count of distinct characters is even or odd.
 * 5. Print the corresponding message.
 *
 * Time Complexity: O(L log L), where L is the length of the string. The dominant
 * operation is the sorting of the string.
 * Space Complexity: O(L) to store the input string.
 */
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
int i, n = 0;
int sum;
string ch;
int main() {
  // Read the user name from standard input.
  cin >> ch;
  /**
   * @brief Sort the string to group identical characters together.
   * This allows for easy counting of unique characters in a single pass.
   */
  sort(ch.begin(), ch.end());
  /**
   * @brief Loop to count the number of distinct characters.
   * Invariant: `n` holds the count of unique characters encountered so far.
   */
  for (i = 0; i < ch.length(); i++) {
    /**
     * @brief If the current character is the same as the next, skip to the next iteration.
     * This avoids double-counting adjacent identical characters.
     */
    if (ch[i] == ch[i + 1]) {
      continue;
    } else {
      /**
       * @brief If the current character is different from the next, it's a new unique character.
       * This also correctly handles the last character of the string, as `ch[ch.length()]`
       * will be a null terminator, different from the last character.
       */
      n++;
    }
  }
  /**
   * @brief Determine the output based on the parity of the distinct character count.
   * Logic: If `n` is even, it's a female user ("CHAT WITH HER!"). If odd, it's a male
   * user ("IGNORE HIM!").
   */
  if (n % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else {
    cout << "IGNORE HIM!" << endl;
  }
  return 0;
}