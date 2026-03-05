/**
 * @file 118A.CPP
 * @brief "String Task" - A string processing problem.
 *
 * This program takes a string as input and processes it according to the following rules:
 * 1. Converts all uppercase letters to lowercase.
 * 2. Deletes all vowels. The vowels are defined as 'a', 'o', 'y', 'e', 'u', 'i'.
 * 3. Inserts a "." character before each consonant.
 * The resulting string is then printed to the console.
 *
 * @algorithm
 * 1. Read the input string `s`.
 * 2. Create an empty string `result` to build the output.
 * 3. Iterate through each character of the input string `s`.
 *    a. Convert the character to lowercase.
 *    b. Check if the character is one of the specified vowels.
 *    c. If it is a vowel, skip to the next character.
 *    d. If it is a consonant, append a '.' character to `result`, followed by the consonant itself.
 * 4. After the loop, print the final `result` string.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the input string, as it involves a single pass.
 * Space Complexity: O(L), to store the input string and potentially O(L) for the result string.
 */
#include <bits/stdc++.h>

using namespace std;

// Forward declaration of the processing function.
void task(string &s);

int main() {
  string s;

  // Read the input string.
  cin >> s;
  // Call the function to process the string.
  task(s);

  return 0;
}

/**
 * @brief Processes the string according to the problem rules.
 * @param s The input string, passed by reference.
 */
void task(string &s) {
  // `result` will store the transformed string.
  string result = {};

  // Loop through each character of the input string.
  for (int i = 0; i < s.size(); ++i) {
    
    // Convert character to lowercase.
    // NOTE: `s[i] < 97` is a simplified check for uppercase, assuming ASCII encoding
    // and that all characters with codes below 'a' are their uppercase equivalents.
    // A more robust method would be `s[i] = tolower(s[i]);`.
    if (s[i] < 97) {
      s[i] += 32;
    }

    // Check if the character is a vowel. The problem defines 'y' as a vowel.
    if (s[i] == 'a' || s[i] == 'o' || s[i] == 'y' || s[i] == 'e' ||
        s[i] == 'u' || s[i] == 'i') {
      // If it's a vowel, ignore it and continue to the next character.
      continue;
    } else {
      // If it's a consonant, prepend a '.' to it in the result string.
      result.push_back('.');
      result.push_back(s[i]);
    }
  }

  // Print the final processed string.
  cout << result;
}
