/**
 * @file 118A.CPP
 * @brief Solution to Codeforces problem 118A - String Task.
 *        This program transforms an input string according to specific rules:
 *        1. All uppercase letters are converted to lowercase.
 *        2. All vowels (a, o, y, e, u, i) are deleted.
 *        3. A '.' character is inserted before each remaining consonant.
 * Algorithm:
 *   1. Reads the input string.
 *   2. Iterates through each character of the input string.
 *   3. For each character:
 *      a. Converts it to lowercase if it's an uppercase letter.
 *      b. Checks if it's a vowel (case-insensitive).
 *      c. If it's a consonant, appends a '.' followed by the consonant to the output string.
 *      d. If it's a vowel, it's skipped (effectively deleted).
 *   4. Prints the resulting transformed string.
 * Time Complexity: O(L * V), where L is the length of the input string and V is the number of vowels
 *                  being checked, due to the nested loop for vowel checking. This could be optimized
 *                  to O(L) by using a hash set or a boolean array for vowel lookup.
 * Space Complexity: O(L) for storing the input and output strings.
 */
#include <cctype>
#include <ctype.h>
#include <iostream>
#include <stdio.h>
#include <string>

using namespace std;

/* 	borrar vocales aeiouy
        agregar un punto antes de una consonante
        reemplazar todas las consonantes mayusculas por minusculas

        -->
        esVocal?
                s , --> borrar
                sino(es consonante) -->
                        agregar un punto antes de esta letra
                        esMayuscula?
                                s  -->	hacerlaMinuscula.

 _________________________

        Programa debe retornar el output como UN s lo string,
         resultando luego de procesar el string

         IP: primera linea, string con longitud de 1 a 100
         OP: imprimir el string resultante, se debe garantizar que este string
 no est  vac o
*/

int main() {
  /**
   * @brief Main function implementing the solution for the "String Task" problem.
   *
   * Reads an input string, processes it character by character to convert
   * uppercase letters to lowercase, remove vowels, and prepend a dot to
   * consonants. Finally, it prints the transformed string.
   * @return int Returns 0 upon successful execution.
   */
  string in, out = ""; // 'in' stores the input string, 'out' builds the transformed string.
  bool isVoel;         // Flag to check if a character is a vowel.
  // Array of vowels (both uppercase and lowercase) for comparison.
  char vowels[] = {'A', 'a', 'E', 'e', 'O', 'o', 'U', 'u', 'I', 'i', 'Y', 'y'};
  getline(cin, in); // Reads the entire line of input into 'in'.

  // Block Logic: Iterates through each character of the input string to apply transformations.
  // Pre-condition: 'in' contains the input string.
  // Invariant: 'out' is progressively built with transformed characters.
  for (int i = 0; i < in.size(); i++) {
    // Block Logic: Converts uppercase characters to lowercase.
    // Pre-condition: 'in[i]' is a character from the input string.
    // Post-condition: If 'in[i]' was uppercase, it becomes lowercase.
    if (in[i] <= 'Z' && in[i] >= 'A')
      in[i] = in[i] - ('Z' - 'z'); // ASCII arithmetic to convert to lowercase.

    isVoel = false; // Reset vowel flag for the current character.
    // Block Logic: Checks if the current character is a vowel.
    // Pre-condition: 'in[i]' is a lowercase character (or other character).
    // Post-condition: 'isVoel' is true if 'in[i]' matches any character in 'vowels'.
    for (int j = 0; j < sizeof(vowels); j++) {
      if (in[i] == vowels[j]) {
        isVoel = true; // Mark as vowel if a match is found.
      }
    }
    // Block Logic: Appends to the output string based on whether the character is a consonant.
    // Pre-condition: 'isVoel' indicates if the character is a vowel.
    // Post-condition: If 'in[i]' is a consonant, it's appended to 'out' preceded by a dot.
    if (isVoel == false) // If the character is not a vowel (i.e., it's a consonant or other non-vowel).
      out = out + '.' + in[i]; // Prepend '.' and append the consonant to the output.
  }
  cout << out << endl; // Prints the final transformed string.

  return 0;
}