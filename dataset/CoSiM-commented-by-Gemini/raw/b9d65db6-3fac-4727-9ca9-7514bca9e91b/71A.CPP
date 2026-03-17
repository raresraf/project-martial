/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem.
 *
 * @details This program reads a number of words and for each word, it checks if its length is greater than 10.
 * If it is, the word is replaced by a special abbreviation. Otherwise, the word is printed as is.
 * The abbreviation consists of the first letter of the word, followed by the number of letters
 * between the first and the last, and finally the last letter.
 *
 * @section
 * Algorithm:
 * 1. Read the number of words, n.
 * 2. Loop n times:
 *    a. Read a word.
 *    b. If the word's length is > 10, print its abbreviation.
 *    c. Otherwise, print the word.
 *
 * Time Complexity: O(N), where N is the number of words to process. The operations on each string (reading, length check, and printing) are proportional to the length of the string, but for the context of this problem, it's considered a fast operation per word.
 * Space Complexity: O(L), where L is the maximum length of a single word to be stored.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  string s;
  cin >> n;
  /**
   * @brief This loop iterates through each of the 'n' test cases.
   * Invariant: The loop will execute exactly 'n' times, processing one word per iteration.
   */
  while (n--) {
    cin >> s;
    /**
     * @brief This block checks if the word is "too long" (length > 10) and applies the abbreviation logic.
     * Pre-condition: The string 's' contains the word for the current test case.
     */
    if (s.length() > 10)
      // If the word is longer than 10 characters, it is abbreviated.
      // The format is: first character, number of middle characters, and last character.
      cout << s.at(0) << s.length() - 2 << s.at(s.length() - 1) << endl;
    else
      // If the word is 10 characters or less, it is printed without modification.
      cout << s << endl;
  }
  return 0;
}
