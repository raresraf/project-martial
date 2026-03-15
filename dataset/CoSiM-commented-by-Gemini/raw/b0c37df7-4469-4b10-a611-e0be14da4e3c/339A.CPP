/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem by extracting, sorting, and reprinting digits.
 * @details This program reads a string representing a sum (e.g., "1+3+2"),
 * extracts the digit characters into a separate array, sorts them, and then
 * prints them back in a sum format.
 *
 * @algorithm
 * 1. Reads the entire sum into a string `s`.
 * 2. Creates a character array `a` to hold the digits.
 * 3. Iterates through the input string `s` with a step of 2 to skip the '+' signs and
 *    copies the digit characters into array `a`.
 * 4. Sorts the array `a` of digit characters using `std::sort`.
 * 5. Prints the elements of the sorted array, placing a '+' between them.
 *
 * @complexity
 * - Time: O(D log D), where D is the number of digits. The complexity is dominated by the sort operation.
 * - Space: O(D) to store the digit characters in the auxiliary array `a`.
 *
 * @note The declaration `char a[s.length() / 2]` uses a Variable Length Array (VLA), a non-standard C++ extension.
 * Furthermore, the size calculation `s.length() / 2` is incorrect for an odd number of digits
 * (e.g., for "1+3+5", length is 5, 3 digits, but size is 2), which can lead to a buffer overflow.
 * The code likely relies on lenient compiler behavior or specific test case constraints to pass.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;

  // Array to store the extracted digit characters.
  // Note: This uses a non-standard VLA and has a potential buffer overflow vulnerability
  // due to incorrect size calculation if the number of digits is `(s.length() / 2) + 1`.
  char a[s.length() / 2];
  
  // k: A counter for the number of digits extracted.
  int k = 0;

  /**
   * @brief This loop extracts digit characters from the input string.
   * It iterates through the string `s` with a step of 2, effectively
   * picking out the characters at even indices (0, 2, 4, ...), which are the digits.
   */
  for (int i = 0; i < s.length(); i = i + 2) {
    a[k++] = s[i];
  }

  // Sorts the array of digit characters. Since ASCII values of '1'...'9' are ordered,
  // this correctly sorts the numbers.
  sort(a, a + k);

  /**
   * @brief This loop reconstructs and prints the sorted sum.
   * It prints each sorted digit, followed by a '+' unless it is the last digit.
   */
  for (int i = 0; i < k; i++) {
    cout << a[i];
    if (i != k - 1)
      cout << "+";
  }
}