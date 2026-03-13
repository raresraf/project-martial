/**
 * @file 118A.CPP
 * @brief Solution for CodeForces Problem 118A - "String Task".
 * 
 * @details
 * This program processes a string according to a specific set of rules:
 * 1. It deletes all vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 * 2. It inserts a character "." before each consonant.
 * 3. It replaces all uppercase consonants with their corresponding lowercase ones.
 *
 * @algorithm String processing
 * @time_complexity O(L), where L is the length of the input string. The program iterates through the string once.
 * @space_complexity O(L) for storing the input and output strings.
 */
#include <iostream>
#include <string>
#include <cctype> // Required for tolower()

using namespace std;

int main() {
  string inp, out = ""; // `inp` for input, `out` for the processed result.
  cin >> inp;

  /**
   * @brief Main loop to iterate through the input string and apply transformation rules.
   * Invariant: `out` contains the processed version of the prefix of `inp` that has been scanned.
   */
  for (int i = 0; i < inp.size(); i++) {
    char lower_char = tolower(inp[i]);
    /**
     * @brief Block to check if the character is a vowel.
     * The character is first converted to lowercase for a simplified comparison.
     * The vowels are 'a', 'e', 'i', 'o', 'u', and 'y'.
     */
    if (lower_char == 'a' || lower_char == 'e' || lower_char == 'i' ||
        lower_char == 'o' || lower_char == 'u' || lower_char == 'y') {
      // If it's a vowel, do nothing and continue to the next character.
      continue;
    } else {
      // If it's a consonant, prepend a '.' and append the lowercase consonant to the output.
      out += ".";
      out += lower_char;
    }
  }

  // Output the final transformed string.
  cout << out << endl;

  return 0;
}
