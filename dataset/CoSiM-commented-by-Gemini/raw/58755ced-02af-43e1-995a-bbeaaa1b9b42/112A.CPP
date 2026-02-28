/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem from Codeforces.
 * @details This problem requires comparing two strings lexicographically, but in a case-insensitive manner.
 * The output should be -1 if the first string is less than the second, 1 if it is greater, and 0 if they are equal.
 *
 * @algorithm
 * 1. Read two strings from standard input.
 * 2. Create two new strings to hold the lowercase versions of the input strings.
 * 3. Iterate through each character of the input strings, convert them to lowercase using `tolower()`,
 *    and append them to their respective new strings.
 * 4. Perform a direct lexicographical comparison on the two new lowercase strings.
 * 5. Print -1, 1, or 0 based on the comparison result. This approach leverages the standard library's
 *    efficient string comparison operators.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the strings. This is dominated by the loops for
 * converting the strings to lowercase and the subsequent comparison.
 * Space Complexity: O(L), as two additional strings are created to store the lowercase versions.
 */
#include <iostream>
#include <string>

int main() {
  /**
   * @variable a, b: The two input strings to be compared.
   * @variable aMale, bMale: Strings to store the lowercase versions of `a` and `b`.
   * The names likely derive from a non-English term for "small letters".
   */
  std::string a, b, aMale, bMale;
  std::cin >> a;
  // This std::cin.get() is functionally inert here as `std::cin >> a` already consumes whitespace.
  std::cin.get();
  std::cin >> b;
  int i = 0;
  char c;

  /**
   * @block
   * @brief Converts the first input string `a` to its lowercase equivalent `aMale`.
   */
  while (a[i]) {
    c = a[i];
    aMale += tolower(c);
    i++;
  }
  i = 0;
  /**
   * @block
   * @brief Converts the second input string `b` to its lowercase equivalent `bMale`.
   */
  while (b[i]) {
    c = b[i];
    bMale += tolower(c);
    i++;
  }

  // The variable `dlugosc` (Polish for "length") is declared but not used in the final logic.
  int dlugosc = aMale.length();

  /**
   * @block
   * @brief Performs case-insensitive lexicographical comparison and prints the result.
   * This is done by comparing the two lowercase strings `aMale` and `bMale`.
   */
  if (aMale < bMale) {
    std::cout << -1; // First string is smaller.
  } else if (aMale > bMale) {
    std::cout << 1;  // First string is larger.
  } else if (aMale == bMale) {
    std::cout << 0;  // Both strings are equal.
  }

  /* This commented-out block contains an alternative, manual implementation of a
   * character-by-character comparison, which is functionally equivalent to the
   * direct string comparison used above.
   *
   * for(i = 0; i < dlugosc; i++){
   *     if(aMale[i] == bMale[i]){
   *         if(i == dlugosc - 1){
   *             std::cout << 0;
   *         }
   *         else
   *         {
   *             continue;
   *         }
   *     }
   *     else
   *     {
   *         if(aMale[i] > bMale[i]){
   *             std::cout << 1;
   *         }
   *         else
   *         {
   *             std::cout << -1;
   *         }
   *     }
   * }
   */

  return 0;
}