/**
 * @file 118A.CPP
 * @brief Solution for the "String Task" problem.
 *
 * @details
 * This program transforms a given string based on a set of rules: it removes all
 * vowels, inserts a '.' before each consonant, and converts all consonants to
 * lowercase. The implementation uses a single-pass, string-building approach.
 * It iterates through the input string and constructs a new result string
 * character by character according to the rules.
 *
 * @section
 * Algorithm: Single-pass string construction.
 * Time Complexity: O(N), where N is the length of the input string.
 * Space Complexity: O(N), to store both the input and the result strings.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // 'word' stores the input string.
  // 'result' will be built up to store the transformed output string.
  string word, result;
  cin >> word;
  /**
   * @brief This loop iterates through the input string to build the new one.
   *
   * Logic: It checks each character of the input 'word'. Vowels are ignored.
   * Consonants are processed by prepending a '.' and converting to lowercase,
   * then appending to the 'result' string.
   */
  for (int i = 0; i < word.length(); i++) {
    // Pre-condition: Check if the character is a vowel (case-insensitive).
    if (word[i] == 'a' || word[i] == 'A' || word[i] == 'e' || word[i] == 'E' ||
        word[i] == 'i' || word[i] == 'I' || word[i] == 'o' || word[i] == 'O' ||
        word[i] == 'u' || word[i] == 'U' || word[i] == 'y' || word[i] == 'Y')
      // If it's a vowel, do nothing and continue to the next character.
      continue;
    else {
      // Invariant: The character is a consonant.
      // Append a '.' to the result.
      result += '.';
      // Convert the consonant to lowercase and append it to the result.
      result += tolower(word[i]);
    }
  }
  // Print the final constructed string.
  cout << result << endl;

  return 0;
}
