/**
 * @5ccc476e-55a3-4ca2-a119-b5288182d2df/281A.CPP
 * @brief This program solves the "Word Capitalization" problem (Codeforces 281A).
 * It reads a single word, ensures its first letter is capitalized (converting it
 * if necessary), and then prints the modified word to standard output.
 *
 * Algorithm: Direct string manipulation - checks and conditionally modifies the first character.
 * Time Complexity: O(L) where L is the length of the input word. Effectively O(1) for typical word lengths.
 * Space Complexity: O(L) for storing the string. Effectively O(1) for typical word lengths.
 */

#include <cstring>  // Included for C-style string functions, though std::string is used.
#include <iostream> // For standard input/output operations (cin, cout).
#include <string>   // For using std::string.
#include <cctype>   // For std::toupper.
using namespace std; // Uses the standard namespace.

int main() {
  string s; // Declares a string variable to store the input word.
  cin >> s; // Reads the word from standard input.
  
  // Inline: Gets the uppercase equivalent of the first character of 's'.
  // This value is used for comparison, not directly assigned initially.
  int upper = toupper(s[0]); 
  
  /**
   * Block Logic: Checks if the first character of the word is already uppercase.
   * If it is, the word is printed as is. Otherwise, the first character is
   * converted to uppercase, and then the modified word is printed.
   * Pre-condition: 's' contains at least one character.
   * Invariant: The word is printed with its first character capitalized.
   */
  if (s[0] == upper) { // If the first character of 's' is already equal to its uppercase form.
    cout << s << '\n'; // Prints the string as is, followed by a newline.
  } else { // If the first character is not uppercase.
    s[0] = upper; // Inline: Converts the first character of 's' to its uppercase equivalent in-place.
    cout << s << '\n'; // Prints the modified string, followed by a newline.
  }
  return 0; // Indicates successful program termination.
}