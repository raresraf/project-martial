/**
 * @file 118A.CPP
 * @brief Implements a string transformation algorithm that converts an input word
 *        to lowercase and then inserts a '.' before each consonant.
 *
 * Algorithm: Iterative character processing. Each character of the input string
 *            is converted to lowercase. It then checks against a predefined
 *            set of vowels. If a character is identified as a consonant, a
 *            '.' is prepended to it in the output string.
 * Time Complexity: O(L * V) where L is the length of the input string and V is
 *                  the number of vowels checked in the inner loop. Given V is
 *                  a small constant (6), this simplifies to O(L).
 * Space Complexity: O(L) for storing the original and the new transformed string.
 */

#include <ctype.h> // Required for tolower()
#include <iostream> // Required for cin, cout
#include <string> // Required for string manipulation
using namespace std;

int main() {
  /**
   * Functional Utility: Reads an input string, processes it character by character
   *                     to convert to lowercase and prepend '.' to consonants,
   *                     then prints the resulting string.
   */
  string palavra; // Stores the input word.
  int contador;   // Used as a flag to check for vowels.

  // Block Logic: Reads a single word (string) from standard input.
  // Pre-condition: Standard input contains a string.
  cin >> palavra;
  string letras = "aeiouy"; // Defines the set of vowels to check against.
  string nova_palavra = "";  // Accumulator for the transformed string.

  /**
   * Block Logic: Iterates through each character of the input string.
   * Pre-condition: `palavra` contains the input string. `nova_palavra` is empty.
   * Invariant: At the start of each iteration, `palavra[i]` is the current character
   *            being processed. `nova_palavra` holds the processed characters up to `i-1`.
   */
  for (int i = 0; i < palavra.size(); i++) {

    // Functional Utility: Converts the current character to its lowercase equivalent.
    palavra[i] = (tolower(palavra[i]));

    contador = 0; // Reset vowel counter for each character.
    /**
     * Block Logic: Checks if the current character is a vowel.
     * Pre-condition: `palavra[i]` is in lowercase. `letras` contains the vowels.
     * Invariant: `contador` will be 1 if `palavra[i]` is a vowel, 0 otherwise.
     */
    for (int j = 0; j < letras.size(); j++) {
      if (palavra[i] == letras[j]) {
        contador += 1; // Increment if a match with a vowel is found.
      }
    }
    // Condition Check: If `contador` remains 0, the character is a consonant.
    if (contador == 0) {
      nova_palavra += ".";      // Prepend a '.' to the consonant.
      nova_palavra += palavra[i]; // Append the consonant itself.
    }
  }

  // Output: Prints the fully transformed string to standard output, followed by a newline.
  cout << nova_palavra << endl;
  return 0; // Indicates successful execution.
}