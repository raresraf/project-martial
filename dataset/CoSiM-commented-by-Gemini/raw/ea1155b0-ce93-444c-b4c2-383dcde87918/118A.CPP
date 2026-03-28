/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces.
 *
 * The program reformats a given string based on three rules:
 * 1. Convert all uppercase letters to their lowercase equivalents.
 * 2. Delete all vowels (a, o, y, e, u, i).
 * 3. Insert a '.' character before each remaining consonant.
 *
 * @algorithm This solution uses idiomatic C++. It first converts the entire
 * input string to lowercase using `std::transform`. Then, it iterates through
 * the modified string, character by character. It checks if a character is a
 * vowel. If not (i.e., it's a consonant), it appends a '.' followed by the
 * character to a new output string.
 *
 * @time_complexity O(N), where N is the length of the input string, due to the
 *                  transform and the loop, which are both linear operations.
 * @space_complexity O(N) to store the resulting output string.
 */
#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string input;
  getline(cin, input);

  // Use `std::transform` to convert the entire string to lowercase.
  transform(input.begin(), input.end(), input.begin(),
            ::tolower);

  string output = "";

  /**
   * @block_logic Iterates through the lowercase string to build the output.
   * Vowels are skipped, and a '.' is prepended to all other characters (consonants).
   */
  for (int i = 0; i < input.length(); i++) {
    if (input.at(i) == 'a' || input.at(i) == 'o' || input.at(i) == 'y' ||
        input.at(i) == 'e' || input.at(i) == 'u' || input.at(i) == 'i') {
      // If the character is a vowel, do nothing.
      continue;
    } else {
      // If the character is a consonant, prepend a '.' and append to the output.
      output = output + "." + input.at(i);
    }
  }

  cout << output;

  return 0;
}
