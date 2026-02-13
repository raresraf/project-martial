/**
 * @file 118A.CPP
 * @brief Solution for the "String Task" problem from competitive programming.
 * @details This program transforms a given string according to a specific set of rules:
 *          1. All uppercase letters are converted to their lowercase counterparts.
 *          2. All vowels are removed. For this problem, the vowels are 'a', 'o', 'y', 'e', 'u', 'i'.
 *          3. A '.' character is inserted before each consonant.
 *          The resulting string is then printed to standard output.
 *
 * @section algorithm Algorithm
 * 1. Read the input string.
 * 2. Create an empty string to store the result.
 * 3. Iterate through each character of the input string:
 *    a. Convert the character to lowercase to handle both cases uniformly.
 *    b. Check if the lowercase character is one of the specified vowels.
 *    c. If it is a vowel, it is effectively deleted by skipping to the next iteration.
 *    d. If it is a consonant, first append a '.' character to the result string,
 *       and then append the consonant itself.
 * 4. After iterating through the entire input string, print the final result.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N), where N is the length of the input string, because the algorithm involves a single pass through the string.
 * - Space Complexity: O(N), as a new string is created to store the output. In the worst case (a string of all consonants), the output string length will be 2*N.
 */
#include <iostream>
#include <string>

/**
 * @brief Prints the final processed string to standard output, followed by a newline.
 * @param v The string to be printed.
 */
void answer(const std::string &v) { std::cout << v << '\n'; }

/**
 * @brief Applies the string transformation rules to the input string.
 * @param input The original string read from standard input.
 */
void solve(const std::string &input) {
  std::string output;
  // Process each character of the input string.
  for (char c : input) {
    c = tolower(c); // Convert character to lowercase.

    // Check if the character is a vowel. The problem defines 'y' as a vowel.
    if (c == 'a' || c == 'o' || c == 'y' || c == 'e' || c == 'u' || c == 'i')
      continue; // If it's a vowel, skip it (effectively deleting it).

    // If the character is a consonant, prepend it with a '.'
    output.push_back('.');
    output.push_back(c);
  }

  answer(output);
}

int main() {
  std::string input;
  std::cin >> input;

  solve(input);

  return 0;
}
