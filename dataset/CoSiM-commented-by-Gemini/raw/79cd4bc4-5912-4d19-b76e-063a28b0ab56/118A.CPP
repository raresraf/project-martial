/**
 * @file 118A.CPP
 * @brief Solution for Codeforces Problem 118A, "String Task".
 * @details The program processes a string based on a set of rules:
 * 1. It deletes all vowels ('A', 'E', 'I', 'O', 'U', 'Y' and their lowercase counterparts).
 * 2. It inserts a '.' character before each consonant.
 * 3. It replaces all uppercase consonants with their corresponding lowercase letter.
 *
 * Algorithm:
 * The code iterates through the input string character by character. For each
 * character, it first converts it to lowercase. Then, it checks if the
 * character is a vowel. If it is not a vowel (i.e., it's a consonant), it
 * appends a '.' followed by the lowercase consonant to the result string.
 * Finally, it prints the fully processed string.
 *
 * Time Complexity: O(N), where N is the length of the input string. The inner
 * loop for vowel checking is constant time (12 iterations).
 * Space Complexity: O(N) to store the output string.
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
  string in, out = "";
  bool isVoel;
  char vowels[] = {'A', 'a', 'E', 'e', 'O', 'o', 'U', 'u', 'I', 'i', 'Y', 'y'};
  getline(cin, in);
  /**
   * @brief Main loop to process each character of the input string.
   */
  for (int i = 0; i < in.size(); i++) {
    /**
     * @brief Pre-processing: Convert the current character to lowercase.
     * This simplifies the subsequent vowel check.
     */
    if (in[i] <= 'Z' && in[i] >= 'A')
      in[i] = in[i] - ('Z' - 'z');
    
    isVoel = false;
    /**
     * @brief Checks if the (now lowercase) character is a vowel by comparing
     * it against a predefined list of vowel characters.
     */
    for (int j = 0; j < sizeof(vowels); j++) {
      if (in[i] == vowels[j]) {
        isVoel = true;
      }
    }

    /**
     * @brief If the character is not a vowel (it's a consonant), prepend a '.'
     * and append it to the output string.
     */
    if (isVoel == false)
      out = out + '.' + in[i];
  }
  cout << out << endl;

  return 0;
}