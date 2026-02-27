/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces.
 * @details The task requires processing a string by applying three rules:
 * 1. Delete all vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 * 2. Insert a character "." before each consonant.
 * 3. Convert all uppercase consonants to lowercase.
 *
 * @algorithm
 * 1. Read the input string.
 * 2. Create an empty result string.
 * 3. Iterate through each character of the input string.
 *    a. Convert the character to lowercase.
 *    b. Check if the lowercase character is a vowel (a, e, i, o, u, y).
 *    c. If the character is not a vowel (i.e., it's a consonant), append a
 *       '.' character to the result string, followed by the lowercase consonant.
 * 4. After iterating through all characters, print the result string.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the input string. The code
 * iterates through the string once, and the inner vowel check is a constant-time
 * operation (since the number of vowels is constant).
 * Space Complexity: O(L) to store the original and the new strings.
 */
#include <ctype.h>
#include <iostream>
#include <string>
using namespace std;

int main() {
  // 'palavra' means 'word' in Portuguese. It stores the input string.
  string palavra;
  // 'contador' means 'counter' in Portuguese. Used as a flag for vowels.
  int contador;

  cin >> palavra;
  // Vowels are defined, including 'y' as per the problem statement.
  // 'letras' means 'letters' in Portuguese.
  string letras = "aeiouy";
  // 'nova_palavra' means 'new word' in Portuguese. This will be the result.
  string nova_palavra = "";

  /**
   * @block
   * @brief Process each character of the input string.
   */
  for (int i = 0; i < palavra.size(); i++) {
    // First, convert the character to lowercase.
    palavra[i] = (tolower(palavra[i]));

    contador = 0; // Reset the vowel flag for the current character.
    /**
     * @block
     * @brief Check if the current character is a vowel.
     */
    for (int j = 0; j < letras.size(); j++) {
      if (palavra[i] == letras[j]) {
        contador += 1; // Set flag if a vowel is found.
      }
    }
    
    /**
     * @brief If the character is not a vowel (it's a consonant).
     */
    if (contador == 0) {
      // Prepend a '.' and append the lowercase consonant to the result.
      nova_palavra += ".";
      nova_palavra += palavra[i];
    }
  }

  cout << nova_palavra << endl;
  return 0;
}
