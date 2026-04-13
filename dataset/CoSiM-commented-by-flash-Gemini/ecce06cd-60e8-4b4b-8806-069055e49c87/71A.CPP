/**
 * @file 71A.CPP
 * @brief This program implements a word abbreviation logic. If a word's length
 *        exceeds 10 characters, it is abbreviated to its first letter, the
 *        count of intermediate characters, and its last letter (e.g., "localization"
 *        becomes "l10n"). Otherwise, the word is printed as is. This implementation
 *        uses `stringstream` for number-to-string conversion and `string::replace`
 *        for in-place modification.
 * Algorithm: String length-based abbreviation, string manipulation.
 * Time Complexity: O(N * L) where N is the number of words and L is the maximum
 *                  length of a word, as string manipulation operations can be proportional
 *                  to the word length.
 * Space Complexity: O(L) for storing the input words and temporary string conversions.
 */
#include <iostream>
#include <sstream> // Required for stringstream operations.
#include <string>

using namespace std;

int main() {
  // Input de # palabras a ingresar // Reads the number of words to input.
  int n; // Number of words.
  cin >> n; // Reads the number of words.

  // Declaro nuevo arreglo de n strings. // Declares a new array of 'n' strings.
  string palabras[n]; // Array to store the input words.

  // Asigno n strings al arreglo palabras. // Assigns 'n' strings to the 'palabras' array.
  /**
   * Block Logic: Reads 'n' words from standard input and stores them in the `palabras` array.
   * Invariant: After each iteration, one word is read and stored in `palabras[i]`.
   */
  for (int i = 0; i < n; i++) {
    string word; // Temporary variable to read each word.
    cin >> word; // Reads a word.
    palabras[i] = word; // Stores the word in the array.
  }

  // Itero sobre cada elemento del arreglo palabras y trabajo con ellos // Iterates over each element of the 'palabras' array and processes it.
  /**
   * Block Logic: Iterates through each word in the `palabras` array. If a word's
   *              length is greater than 10, it's abbreviated. Otherwise, it's printed as is.
   * Invariant: Each word is processed according to the abbreviation rules.
   */
  for (int j = 0; j < n; j++) {
    if (palabras[j].length() > 10) { // Checks if the word's length is greater than 10.

      // Cuento # letras entre la primera y la  ltima. // Counts the number of letters between the first and last.
      int count = 0; // Counter for intermediate letters.
      /**
       * Block Logic: Calculates the number of characters between the first and last characters of the word.
       * Invariant: `count` stores `word.length() - 2`.
       */
      for (int k = 1; k < palabras[j].length() - 1; k++) {
        count++; // Increments the counter for each intermediate character.
      }

      // Convierto esta cantidad a un string // Converts this quantity to a string.
      string result;      // String to store the converted count.
      stringstream sstm;  // Creates a stringstream object.
      sstm << count;      // Inserts the integer count into the stringstream.
      result = sstm.str(); // Extracts the string representation of the count.

      // Modifico la palabra en la posicion j-esima por 'AcountZ' // Modifies the word at j-th position to 'AcountZ'.
      // This line replaces the middle part of the word with the `result` string (the count).
      palabras[j] = palabras[j].replace(palabras[j].begin() + 1,
                                        palabras[j].end() - 1, result);
      cout << palabras[j] << endl; // Prints the abbreviated word.
    } else {
      cout << palabras[j] << endl; // Prints the original word if not abbreviated.
    }
  }

  return 0;
}