/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces (Problem 118A).
 *        This program takes an input string, converts all its uppercase letters
 *        to lowercase, removes all vowels (a, o, y, e, u, i), and then inserts
 *        a '.' character before each remaining consonant. The modified string is
 *        then printed.
 * Algorithm: String Manipulation / Character Processing
 * Time Complexity: O(L), where L is the length of the input string. Each character
 *                  is processed a constant number of times. Appending characters
 *                  to `std::string` is amortized O(1) per character.
 * Space Complexity: O(L) to store the input string and the resulting modified string.
 */

#include <iostream> // Required for input/output operations (cin, cout).
#include <string>   // Required for using std::string.
#include <cctype>   // Required for character functions like tolower (though custom implementation is used).

using namespace std;

/**
 * @brief Main function to process a string according to the "String Task" rules.
 * @return 0 if the program executes successfully.
 */
int main() {
  string s, r; // Declare 's' for the input string and 'r' for the result string.
  // Read the input string.
  cin >> s;
  
  /**
   * Block Logic: Iterates through each character of the input string 's'.
   * Invariant: 'r' accumulates the processed characters with '.' prepended to consonants.
   */
  for (int i = 0; i < s.size(); i++) {
    char c = s[i]; // Get the current character.
    
    /**
     * Block Logic: Converts uppercase letters to lowercase.
     * Pre-condition: 'c' is the current character from the input string.
     * Post-condition: 'c' is converted to lowercase if it was uppercase.
     */
    if (c >= 'A' && c <= 'Z') {
      // Inline: Convert uppercase 'c' to lowercase by adjusting its ASCII value.
      // 'a' - 'A' gives the difference in ASCII values between lowercase and uppercase.
      c += 'a' - 'A';
      // c+= 32; // Alternative (commented out): direct ASCII addition, valid if character set is ASCII.
    }
    
    /**
     * Block Logic: Checks if the character is a consonant and appends it to the result string 'r'
     *              with a preceding dot.
     * Pre-condition: 'c' is a lowercase character.
     * Post-condition: 'r' is modified by appending '.' and 'c' if 'c' is a consonant.
     */
    if (c != 'a' && c != 'e' && c != 'i' && c != 'o' && c != 'u' && c != 'y') {
      r.append(1, '.'); // Append a dot character.
      r.append(1, c);   // Append the consonant character.
    }
  }
  // Print the final modified string 'r' followed by a newline.
  cout << r << endl;
  return 0; // Indicate successful program termination.
}
