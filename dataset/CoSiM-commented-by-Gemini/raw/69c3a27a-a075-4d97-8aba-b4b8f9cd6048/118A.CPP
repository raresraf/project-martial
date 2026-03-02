// CodeForces
// 118A - StringTask

#include <iostream>
using namespace std;

/**
 * @brief Checks if a character is a vowel.
 * @param c The character to check.
 * @return True if the character is a vowel (a, e, i, o, u, y), case-insensitively.
 * False otherwise.
 */
bool isVowel(char c) {
  c = (char)tolower(c);

  if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y')
    return true;
  else
    return false;
}

/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem from Codeforces (Problem 118A).
 * @details The program processes a string by deleting vowels, inserting a '.' before
 * each consonant, and converting all consonants to lowercase.
 *
 * Algorithm:
 * The program iterates through each character of an input string. For each character,
 * it first checks if it is a vowel (case-insensitively, including 'y').
 * - If the character is a vowel, it is ignored.
 * - If the character is a consonant, a '.' character is appended to the result string,
 *   followed by the lowercase version of the consonant.
 * The final constructed string is then printed.
 *
 * Time Complexity: O(L), where L is the length of the input string, due to the single
 * pass through the string and string appends.
 * Space Complexity: O(L), to store the input and output strings.
 */
int main() {
  string word;
  cin >> word;

  string output("");

  /**
   * @brief Main loop to process the string and build the output.
   */
  for (unsigned int i = 0; i < word.size(); i++) {
    // If the character is not a vowel (i.e., it's a consonant)...
    if (!isVowel(word[i])) {
      // ...append a '.' and the lowercase consonant to the output.
      output = output + "." + (char)tolower(word[i]);
    }
  }

  cout << output;

  return 0;
}
