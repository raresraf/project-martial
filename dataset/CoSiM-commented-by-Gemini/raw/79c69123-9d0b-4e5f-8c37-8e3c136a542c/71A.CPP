/**
 * @file 71A.CPP
 * @brief An attempted solution for Codeforces Problem 71A, "Way Too Long Words".
 * @details This program is intended to shorten words that are "too long" by
 * replacing the middle characters with a number representing their count.
 *
 * Flawed Algorithm:
 * 1. Reads a number of words `t`.
 * 2. For each word, it constructs an "abbreviated" version.
 *    - The abbreviation is formed by the first character, the string's TOTAL length,
 *      and the last character. This is incorrect; it should be the length of the
 *      middle part (total length - 2).
 * 3. It checks if the word's length is 10 or more. This is also incorrect; the
 *    problem specifies words with a length STRICTLY GREATER than 10.
 * 4. If the length condition is met, the incorrectly abbreviated word is stored.
 * 5. Words that do not meet the length condition are discarded and never printed.
 * 6. Finally, it prints only the stored (abbreviated) words.
 *
 * Time Complexity: O(T * L), where T is the number of test cases and L is the
 * length of the string, due to string operations.
 * Space Complexity: O(T * L) to store the results in a vector.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int t;
  cin >> t;
  vector<string> use_case;

  /**
   * @brief This loop iterates through each word provided in the input.
   */
  for (int i = 0; i < t; i++) {
    string h, c1, c3;
    cin >> h;
    
    /**
     * @brief This block constructs an abbreviated version of the string `h`.
     * Note: The logic is flawed. It uses the total length of the string `h` for
     * the numerical part, not the length of the characters between the first
     * and last, as required by the problem. For a 12-char word, it would
     * incorrectly produce "l12n" instead of "l10n".
     */
    c1 = h[0];
    int j = h.length();
    std::string c2 = std::to_string(j);
    c3 = h[j - 1];
    c2.append(c3);
    c1.append(c2);
    
    /**
     * @brief This condition checks if the word should be abbreviated.
     * Note: This check is flawed in two ways:
     * 1. It uses `>= 10`, while the problem requires a length strictly `> 10`.
     * 2. The `else` block is missing, so words that are not "too long" are
     *    never added to the `use_case` vector and are effectively lost.
     */
    if (h.length() >= 10) {
      use_case.push_back(c1);
    }
  }

  /**
   * @brief This loop prints the results. It will only print the abbreviated
   * words that were stored, not the original short words.
   */
  for (int i = 0; i < use_case.size(); i++) {
    cout << use_case[i] << endl;
  }
  return 0;
}
