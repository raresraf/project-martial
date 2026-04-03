/**
 * @8c942a1d-3b2a-4a02-8c83-bdbcc7b7a161/118A.CPP
 * @brief This program processes an input string by removing all vowels (case-insensitive),
 * converting any remaining uppercase consonants to lowercase, and then printing each
 * consonant prefixed with a dot. This is a common string manipulation problem.
 * Algorithm: Character-by-character filtering and transformation.
 * Time Complexity: O(N), where N is the length of the input string, due to a single linear pass.
 * Space Complexity: O(N) to store the input string, plus O(1) for auxiliary variables.
 */

#include <cctype>     // Required for isupper, tolower functions
#include <iostream>   // Required for standard input/output operations (cin, cout)
#include <string>     // Required for std::string

using namespace std;

int main() {
  // @var test: Stores the input string from the user.
  string test;
  cin >> test; // Read the input string.
  
  // Block Logic: Iterate through each character of the input string.
  // Invariant: Characters are processed from left to right.
  // Pre-condition: 'test' contains the string to be processed.
  for (int i = 0; i < test.length(); i++) {
    // Block Logic: Check if the current character is a vowel (case-insensitive).
    // If it is a vowel, skip to the next character (effectively removing it).
    if (test[i] == 'A' || test[i] == 'O' || test[i] == 'Y' || test[i] == 'E' ||
        test[i] == 'U' || test[i] == 'I' || test[i] == 'a' || test[i] == 'o' ||
        test[i] == 'y' || test[i] == 'e' || test[i] == 'u' || test[i] == 'i') {
      continue; // Skip vowels.
    } 
    // Block Logic: If the character is not a vowel, it's a consonant. Process it.
    else {
      // Block Logic: Check if the consonant is uppercase.
      if (isupper(test[i])) {
        // Inline: Convert the uppercase consonant to lowercase.
        test[i] = tolower(test[i]);
        // Inline: Print the lowercase consonant prefixed with a dot.
        cout << "." << test[i];
      } 
      // Block Logic: If the consonant is already lowercase, print it as is.
      else {
        // Inline: Print the lowercase consonant prefixed with a dot.
        cout << "." << test[i];
      }
    }
  }
  // Post-condition: No newline is printed implicitly; the problem usually expects output on one line.
  // However, for standard competitive programming, a final newline is usually required.
  // The current code doesn't explicitly add one, which might be an oversight for some platforms.
  return 0; // Explicitly return 0 for successful execution.
}
