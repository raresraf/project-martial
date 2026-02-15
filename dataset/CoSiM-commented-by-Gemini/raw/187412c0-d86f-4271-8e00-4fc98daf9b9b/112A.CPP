/**
 * @file 112A.CPP
 * @brief Solution to the "Petya and Strings" competitive programming problem.
 * @details This program performs a case-insensitive lexicographical comparison of two input strings
 * and outputs -1, 0, or 1 based on the comparison result.
 *
 * @section algorithm Algorithm
 * The algorithm normalizes the strings to a single case (lowercase) and then compares them:
 * 1. Read the two strings, `palabra1` and `palabra2`.
 * 2. Iterate through each character of both strings.
 * 3. In the loop, it checks if a character's ASCII value is less than 92. This is an
 *    unconventional way to identify uppercase English letters (ASCII for 'A'-'Z' is 65-90).
 * 4. If a character is determined to be uppercase, it's converted to lowercase by adding 32
 *    to its ASCII value, which is the standard ASCII difference between cases.
 * 5. After the loop finishes, both strings are guaranteed to be in lowercase.
 * 6. A standard lexicographical comparison is performed on the normalized strings.
 * 7. A result variable is set to 1 if the first string is greater, -1 if it is smaller,
 *    and remains 0 if they are equal.
 * 8. Finally, the result is printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(L), where L is the length of the strings. The program iterates
 *   through the strings once for case conversion, and the final comparison also takes O(L).
 * - Space Complexity: O(L) to store the two input strings.
 */
#include <iostream>
#include <string>

// Note: Using standard library functions like `tolower()` from `<cctype>`
// would be a more readable and portable way to handle case conversion.

using namespace std;

int main() {
  // `palabra1` and `palabra2` are Spanish for "word1" and "word2".
  string palabra1, palabra2;
  cin >> palabra1;
  cin >> palabra2;
  
  // `resultado` is Spanish for "result". It's initialized to 0, the value for equality.
  int resultado = 0;

  // Loop through the strings to convert all characters to lowercase for case-insensitive comparison.
  for (int i = 0; i < palabra1.size(); i++) {
    // The ASCII values for 'A'-'Z' are 65-90. Checking for < 92 is a way to identify
    // uppercase letters, assuming the input is only English alphabet characters.
    if (palabra1[i] < 92) {
      // Convert to lowercase by adding 32, the ASCII difference between 'a' and 'A'.
      palabra1[i] += 32;
    }
    if (palabra2[i] < 92) {
      palabra2[i] += 32;
    }
  }

  // Perform lexicographical comparison on the now-lowercase strings.
  if (palabra1 > palabra2)
    resultado = 1;
  else if (palabra1 < palabra2)
    resultado = -1;
  
  // Output the result: -1, 0, or 1.
  cout << resultado;
}
