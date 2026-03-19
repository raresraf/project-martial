/**
 * @file 118A.CPP
 * @brief Solution to Codeforces Problem 118A: "String Task".
 *
 * This problem involves transforming a given string according to specific rules:
 * 1. All uppercase letters should be converted to lowercase.
 * 2. Vowels ('a', 'o', 'y', 'e', 'u', 'i') should be deleted.
 * 3. A "." should be inserted before each consonant.
 *
 * Algorithm: String iteration and character processing.
 * The algorithm iterates through the input string character by character. For each character,
 * it first converts it to lowercase if it's uppercase. Then, it checks if the character
 * is a vowel. If it's a consonant, a "." is prepended to it in the result string.
 *
 * Time Complexity: O(N), where N is the length of the input string, as it involves
 *                  a single pass through the string with constant time operations per character.
 * Space Complexity: O(N) for storing the result string.
 */
#include <bits/stdc++.h> // Includes various standard libraries, including iostream and string.

using namespace std;

void task(string &s); // Function prototype for the string transformation task.

/**
 * @brief Main function to execute the string transformation.
 * Reads the input string and calls the 'task' function to perform the transformation.
 *
 * @return int Returns 0 on successful execution.
 */
int main() {
  string s; // Declares a string to store the input.

  cin >> s; // Reads the input string.
  task(s);  // Calls the task function to process the string.

  return 0; // Indicates successful program execution.
}

/**
 * @brief Transforms the given string according to the problem rules.
 * Converts uppercase letters to lowercase, removes vowels, and inserts '.'
 * before each consonant. The modified string is then printed to standard output.
 *
 * @param s A reference to the string to be transformed.
 *          The string is modified in place for lowercase conversion,
 *          but the result is built into a new string.
 */
void task(string &s) {
  string result = {}; // Initializes an empty string to build the transformed result.

  /**
   * Block Logic: Iterates through each character of the input string to apply transformation rules.
   * Pre-condition: 's' contains the original string.
   * Invariant: 'result' accumulates the transformed characters.
   */
  for (int i = 0; i < s.size(); ++i) {
    // Block Logic: Converts uppercase characters to lowercase.
    // Explanation: ASCII values for 'A'-'Z' are less than 'a'-'z'. Adding 32 converts uppercase to lowercase.
    if (s[i] < 97) {
      s[i] += 32;
    }

    // Block Logic: Checks if the character is a vowel.
    // If it's a vowel, it's skipped (effectively deleted).
    if (s[i] == 'a' || s[i] == 'o' || s[i] == 'y' || s[i] == 'e' ||
        s[i] == 'u' || s[i] == 'i') {
      continue; // Skips to the next character if it's a vowel.
    } else {
      // Block Logic: If the character is a consonant, a '.' is prepended, and then the consonant itself is added.
      result.push_back('.'); // Appends a dot.
      result.push_back(s[i]); // Appends the consonant.
    }
  }

  cout << result; // Prints the final transformed string.
}