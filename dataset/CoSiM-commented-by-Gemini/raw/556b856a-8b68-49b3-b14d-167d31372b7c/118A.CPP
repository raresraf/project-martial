/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces.
 * @details The task is to process a string by deleting all vowels, inserting a
 * '.' before each consonant, and converting all consonants to lowercase.
 *
 * @algorithm
 * This solution iterates through the input string and builds a new output string
 * based on the rules.
 * 1. Read the input word.
 * 2. Create an empty string for the output.
 * 3. Loop through each character of the input word.
 * 4. For each character, check if it is a vowel (both uppercase and lowercase)
 *    by comparing it against its ASCII value.
 * 5. If the character is a vowel, do nothing (effectively deleting it).
 * 6. If the character is not a vowel (it's a consonant), append a '.' to the
 *    output string, followed by the lowercase version of the consonant.
 * 7. After the loop, print the final output string.
 *
 * @note This implementation uses explicit ASCII values for vowel checking, which
 * is less readable than using character literals.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the string, due to the single pass.
 * Space Complexity: O(L) to store both the input and output strings.
 */
#include <cctype>
#include <iostream>
#include <string>

int main() {
  // 'slowo' is Polish for "word".
  std::string slowo;
  std::cin >> slowo;
  
  // 'dlugosc' is Polish for "length".
  int dlugosc = slowo.length();
  std::string output = "";
  char c;

  /**
   * @block
   * @brief Process each character of the input string.
   */
  for (int i = 0; i < dlugosc; i++) {
    // Check if the character is an uppercase vowel using ASCII values.
    // 'A', 'O', 'Y', 'E', 'U', 'I'
    if (slowo[i] == 65 || slowo[i] == 79 || slowo[i] == 89 || slowo[i] == 69 ||
        slowo[i] == 85 || slowo[i] == 73) {
      continue; // Skip vowels.
    } 
    // Check if the character is a lowercase vowel using ASCII values.
    // 'a', 'o', 'y', 'e', 'u', 'i'
    else if (slowo[i] == 97 || slowo[i] == 111 || slowo[i] == 121 ||
               slowo[i] == 101 || slowo[i] == 117 || slowo[i] == 105) {
      continue; // Skip vowels.
    } 
    // If the character is not a vowel, it's a consonant.
    else {
      output += ".";
      c = slowo[i];
      output += tolower(c); // Append '.' and the lowercase consonant.
    }
  }

  std::cout << output;

  return 0;
}
