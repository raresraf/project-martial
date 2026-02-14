/**
 * @file 112A.CPP
 * @brief Solution for a problem requiring case-insensitive lexicographical string comparison.
 * @details This program compares two input strings lexicographically, ignoring case differences.
 * It outputs -1 if the first string is smaller, 1 if it's larger, and 0 if they are equal.
 * The variable names are in Spanish.
 *
 * @section algorithm Algorithm
 * 1. Read two strings, `palabra1` (word1) and `palabra2` (word2).
 * 2. Convert both strings to lowercase. This is achieved by iterating through each character
 *    and checking if its ASCII value is less than 92 (which covers all uppercase English letters).
 *    If it is, 32 is added to the ASCII value to get the lowercase equivalent.
 * 3. After conversion, the two strings are compared lexicographically.
 * 4. The `resultado` (result) is set based on the comparison:
 *    - `1` if `palabra1` is greater than `palabra2`.
 *    - `-1` if `palabra1` is less than `palabra2`.
 *    - `0` if they are equal (the initial value).
 * 5. The final `resultado` is printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(L), where L is the length of the strings. The process involves a single pass
 *   for case conversion and another for comparison.
 * - Space Complexity: O(L) to store the two input strings.
 */
#include <iostream>
#include <string>
#include <cctype>

using namespace std;

/**
 * @brief Main function to perform the case-insensitive string comparison.
 * @return 0 on successful execution.
 */
int main() {
  string palabra1, palabra2; // Spanish for "word1" and "word2"
  cin >> palabra1;
  cin >> palabra2;
  
  int resultado = 0; // Spanish for "result"
  
  // Loop through the strings to convert them to lowercase.
  for (int i = 0; i < palabra1.size(); i++) {
    palabra1[i] = tolower(palabra1[i]);
    palabra2[i] = tolower(palabra2[i]);
  }
  
  // Perform the lexicographical comparison.
  if (palabra1 > palabra2) {
    resultado = 1;
  } else if (palabra1 < palabra2) {
    resultado = -1;
  }
  
  // Print the result.
  cout << resultado;
  
  return 0;
}
