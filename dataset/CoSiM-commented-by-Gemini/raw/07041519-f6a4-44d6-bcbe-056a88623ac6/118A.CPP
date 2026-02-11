/**
 * @file 118A.CPP
 * @brief Processes a string by removing vowels and adding a '.' before consonants.
 * @details This program implements the "String Task" from Codeforces. It takes a string,
 * converts all its characters to lowercase, removes all vowels (a, o, y, e, u, i),
 * and for each remaining character (consonant), it prints a '.' followed by that consonant.
 *
 * @see https://codeforces.com/problemset/problem/118/A
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Iterate through each character of the string.
 * 3. Convert the character to lowercase.
 * 4. Check if the character is a vowel ('a', 'e', 'i', 'o', 'u', 'y').
 * 5. If it is not a vowel (i.e., it's a consonant), print a '.' character followed by the consonant itself.
 * 6. Vowels are skipped, effectively removing them from the output.
 *
 * Time Complexity: O(N), where N is the length of the input string, because it involves a single pass through the string.
 * Space Complexity: O(N) to store the input string.
 */

#include <iostream>
#include <string> // Recommended to include for std::string usage
#include <cctype>   // Recommended to include for tolower

using namespace std;

int main() {
  // The input string.
  string word;
  int c, i;

  cin >> word;
  c = word.size();

  /**
   * @brief Main loop to process each character of the string.
   *
   * Pre-condition: 'word' contains the input string.
   * Invariant: The first 'i' characters of the word have been processed and the resulting
   *            consonant-punctuated string has been printed.
   */
  for (i = 0; i < c; i++) {
    // Convert the character to lowercase for case-insensitive vowel checking.
    word[i] = char(tolower(word[i]));

    // Block Logic: Checks if the character is a consonant.
    // The vowels are 'a', 'e', 'i', 'o', 'u', and for this problem, 'y'.
    if (word[i] != 'a' && word[i] != 'e' && word[i] != 'i' && word[i] != 'o' &&
        word[i] != 'u' && word[i] != 'y')
      // If it's a consonant, print a '.' followed by the character.
      cout << '.' << word[i];
  }

  return 0;
}
