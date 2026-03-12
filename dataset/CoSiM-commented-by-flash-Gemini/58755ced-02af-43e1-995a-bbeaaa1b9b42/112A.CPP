/**
 * @58755ced-02af-43e1-995a-bbeaaa1b9b42/112A.CPP
 * @brief This program performs a case-insensitive lexicographical comparison of two input strings.
 * It first converts both strings to their lowercase equivalents and then uses standard string
 * comparison to determine their relative order.
 *
 * Algorithm: String conversion to lowercase followed by direct lexicographical comparison.
 * Time Complexity: O(L) where L is the maximum length of the input strings.
 * Space Complexity: O(L) for storing the lowercase versions of the strings.
 */

#include <iostream> // For standard input/output operations (std::cin, std::cout).
#include <string>   // For using std::string.
#include <cctype>   // For character handling functions like std::tolower.

int main() {
  std::string a, b, aMale, bMale; // Declares strings for input (a, b) and their lowercase versions (aMale, bMale).
  std::cin >> a; // Reads the first string from standard input.
  // std::cin.get(); // Consumes the newline character left in the buffer after reading 'a'.
                      // This line is often used when switching between formatted input (like `>>`)
                      // and line-based input (`getline`), but might be vestigial here.
  std::cin >> b; // Reads the second string from standard input.
  int i = 0;     // Loop counter and index for character processing.
  char c;        // Temporary character variable.

  /**
   * Block Logic: Converts the first input string 'a' to its lowercase equivalent and stores it in 'aMale'.
   * Invariant: 'aMale' contains the lowercase version of 'a'.
   * Pre-condition: 'a' is a valid string.
   */
  while (a[i]) { // Iterates through string 'a' until the null terminator is reached (C-style string check).
    c = a[i];    // Inline: Gets the character at the current index.
    aMale += tolower(c); // Inline: Converts the character to lowercase and appends it to 'aMale'.
    i++;         // Increments the index.
  }
  i = 0; // Resets the loop counter for the second string.
  /**
   * Block Logic: Converts the second input string 'b' to its lowercase equivalent and stores it in 'bMale'.
   * Invariant: 'bMale' contains the lowercase version of 'b'.
   * Pre-condition: 'b' is a valid string.
   */
  while (b[i]) { // Iterates through string 'b' until the null terminator is reached.
    c = b[i];    // Inline: Gets the character at the current index.
    bMale += tolower(c); // Inline: Converts the character to lowercase and appends it to 'bMale'.
    i++;         // Increments the index.
  }

  // int dlugosc = aMale.length(); // Declared but not used in the final logic.

  /**
   * Block Logic: Compares the two lowercase strings lexicographically.
   * Pre-condition: 'aMale' and 'bMale' contain the lowercase versions of the input strings.
   * Invariant: One of -1, 0, or 1 is printed based on the comparison result.
   */
  if (aMale < bMale) { // If aMale is lexicographically smaller than bMale.
    std::cout << -1;   // Prints -1.
  } else if (aMale > bMale) { // If aMale is lexicographically greater than bMale.
    std::cout << 1;    // Prints 1.
  } else if (aMale == bMale) { // If aMale is lexicographically equal to bMale.
    std::cout << 0;    // Prints 0.
  }

  return 0; // Indicates successful program termination.
}