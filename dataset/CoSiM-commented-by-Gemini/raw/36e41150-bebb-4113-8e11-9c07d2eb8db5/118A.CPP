/**
 * @file 118A.CPP
 * @brief An attempt to solve the "String Task" problem.
 *
 * This program processes a string based on a set of rules: delete vowels,
 * insert a '.' before each consonant, and convert all consonants to lowercase.
 *
 * The intended algorithm is:
 * 1. Read an input string.
 * 2. Create a new result string.
 * 3. Iterate through the input string, character by character.
 * 4. Convert the character to lowercase.
 * 5. If the character is a consonant, append a '.' and then the character to the result.
 * 6. If it's a vowel, do nothing.
 * 7. Print the final result string.
 *
 * @note This implementation is buggy. It processes the string in a single pass
 * but fails to handle uppercase consonants correctly. It also prints directly
 * to the console instead of building a new string.
 *
 * Time Complexity: O(L), where L is the length of the string.
 * Space Complexity: O(L) to store the input string.
 */
#include <iostream>
#include <string> // For std::string
#include <cctype> // For tolower

using namespace std;
string a; // Global string variable to hold the input.

int main() {
  cin >> a;
  
  /**
   * Block Logic: Main processing loop.
   * Iterates through each character of the input string 'a'.
   */
  for (int i = 0; i < a.length(); i++) {
    /**
     * Case 1: Character is a vowel (uppercase or lowercase).
     * The code correctly identifies vowels but the action `cout << ""`
     * is a no-op, effectively deleting the vowel from the output stream.
     */
    if (a[i] == 'A' || a[i] == 'O' || a[i] == 'Y' || a[i] == 'E' ||
        a[i] == 'U' || a[i] == 'I' || a[i] == 'a' || a[i] == 'o' ||
        a[i] == 'y' || a[i] == 'e' || a[i] == 'u' || a[i] == 'i') {
      cout << "";
    } 
    /**
     * Case 2: Character is an uppercase consonant.
     * @note Bug: The code correctly converts the character to lowercase in-place,
     * but then fails to print it or the required '.' prefix. This block
     * effectively deletes uppercase consonants from the output.
     */
    else if (a[i] == 'B' || a[i] == 'C' || a[i] == 'D' || a[i] == 'F' ||
               a[i] == 'G' || a[i] == 'H' || a[i] == 'J' || a[i] == 'K' ||
               a[i] == 'L' || a[i] == 'M' || a[i] == 'N' || a[i] == 'P' ||
               a[i] == 'Q' || a[i] == 'R' || a[i] == 'S' || a[i] == 'T' ||
               a[i] == 'W' || a[i] == 'V' || a[i] == 'X' || a[i] == 'Z') {
      a[i] = (char)tolower(a[i]);

    } 
    /**
     * Case 3: Character is a lowercase consonant.
     * This is the only case that produces the correct output format: a '.'
     * followed by the lowercase consonant.
     */
    else if (a[i] == 'b' || a[i] == 'c' || a[i] == 'd' || a[i] == 'f' ||
               a[i] == 'g' || a[i] == 'h' || a[i] == 'j' || a[i] == 'k' ||
               a[i] == 'l' || a[i] == 'm' || a[i] == 'n' || a[i] == 'p' ||
               a[i] == 'q' || a[i] == 'r' || a[i] == 's' || a[i] == 't' ||
               a[i] == 'w' || a[i] == 'v' || a[i] == 'x' || a[i] == 'z') {
      cout << "." << a[i];
    }
  }
}
