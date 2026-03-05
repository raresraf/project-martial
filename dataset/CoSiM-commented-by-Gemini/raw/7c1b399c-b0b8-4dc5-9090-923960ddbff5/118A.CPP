// 4005085     Jul 3, 2013 3:50:19 AM	fuwutu	 118A - String Task
// GNU C++0x	Accepted	15 ms	0 KB

/**
 * @file 118A.CPP
 * @brief Solution for Codeforces Problem 118A, "String Task".
 * @details The program transforms a given string by the following rules:
 * 1. Deletes all vowels ('a', 'e', 'i', 'o', 'u', 'y').
 * 2. Inserts a '.' character before each consonant.
 * 3. Converts all uppercase consonants to their lowercase equivalents.
 *
 * Algorithm:
 * The code iterates through each character of the input string. For each
 * character, it first converts it to lowercase. Then, it checks if the
 * character is a vowel. If it's not a vowel, the character is considered a
 * consonant. A '.' character and the lowercase consonant are then appended to
 * a result string. Finally, the result string is printed.
 *
 * Time Complexity: O(N), where N is the length of the input string.
 * Space Complexity: O(N) to store the resulting string.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s, r;
  cin >> s;

  /**
   * @brief Iterates through each character of the input string `s`.
   */
  for (char c : s) {
    /**
     * @brief Converts the character to lowercase to simplify processing.
     */
    if (c >= 'A' && c <= 'Z') {
      c += 'a' - 'A';
    }

    /**
     * @brief Checks if the character is a consonant by testing if it is not
     * one of the vowels.
     */
    if (c != 'a' && c != 'e' && c != 'i' && c != 'o' && c != 'u' && c != 'y') {
      // If it's a consonant, prepend a '.' and append to the result string.
      r.append(1, '.');
      r.append(1, c);
    }
  }
  cout << r << endl;
  return 0;
}
