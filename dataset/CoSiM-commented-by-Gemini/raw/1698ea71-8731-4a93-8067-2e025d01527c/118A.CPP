/**
 * @file 118A.CPP
 * @brief Solution for Contest 118, Problem A: String Task.
 * @author Rakesh Kumar
 * @date 17/01/2020
 *
 * @details This program transforms a string based on a set of rules: it removes all
 * vowels, converts all remaining characters (consonants) to lowercase, and prepends
 * a '.' character to each consonant.
 *
 * @section algorithm Algorithm
 * 1. Read the entire input line into a string.
 * 2. Create an empty string to build the result.
 * 3. Iterate through each character of the input string.
 * 4. For each character, use a helper function `vowel()` to check if it is a vowel
 *    (case-insensitively). The vowels are defined as 'a', 'o', 'y', 'e', 'u', 'i'.
 * 5. If the character is not a vowel (i.e., it's a consonant):
 *    a. Convert the consonant to its lowercase equivalent.
 *    b. Append a '.' character to the result string.
 *    c. Append the lowercase consonant to the result string.
 * 6. After iterating through the entire input string, print the final result.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N), where N is the length of the input string, as it requires a single pass.
 * - Space Complexity: O(N), for storing the result string, which can be up to 2*N in length in the worst case (all consonants).
 */
#include <bits/stdc++.h>

/**
 * @brief Checks if a character is a vowel, case-insensitively.
 * @param c The character to check.
 * @return True if the character is one of 'a', 'o', 'y', 'e', 'u', 'i' (in upper or lower case), false otherwise.
 */
inline bool vowel(char c) {
  if (c == 'a' || c == 'A' || c == 'o' || c == 'O' || c == 'y' || c == 'Y' ||
      c == 'e' || c == 'E' || c == 'u' || c == 'U' || c == 'i' || c == 'I')
    return true;
  return false;
}

int main() {
  std::string str;
  std::getline(std::cin, str);

  std::string result;
  // Iterate through each character of the input string.
  for (char c : str) {
    // Process the character only if it is not a vowel.
    if (!vowel(c)) {
      c = std::tolower(c); // Convert the consonant to lowercase.
      result += ".";           // Prepend a dot.
      result.push_back(c);   // Append the lowercase consonant.
    }
  }
  std::cout << result << std::endl;

  return 0;
}
