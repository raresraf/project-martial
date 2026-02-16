/**
 * @file 71A.CPP
 * @brief Solution for a competitive programming problem that abbreviates long words.
 *
 * Algorithm: Processes multiple test cases, for each word checks its length. If length > 10,
 *            it prints an abbreviation (first char + (length-2) + last char); otherwise, it prints the word as is.
 * Time Complexity: O(T * L) where T is the number of test cases and L is the average length of a word.
 *                  The `strlength` function is O(L), and the loop runs T times.
 * Space Complexity: O(L_max) for storing the longest word, where L_max is the maximum possible word length.
 */

#include <iostream>
#include <string>
using namespace std;
/**
 * @brief Computes the length of a given string.
 *
 * Functional Utility: This function manually calculates the length of a null-terminated
 *                     string by iterating through its characters until the null
 *                     terminator (`\0`) is encountered.
 * Time Complexity: O(L) where L is the length of the input string `a`.
 * Space Complexity: O(1) as it uses a constant amount of extra space.
 *
 * @param a The input string whose length is to be calculated.
 * @return The length of the string `a`.
 */
int strlength(string a) {
  int count = 0;
  // Block Logic: Iterate through the string until the null terminator is found.
  // Invariant: `count` is incremented for each character until the end of the string.
  while (a[count] != '\0') {
    count++;
  }
  return count;
}
/**
 * @brief Main entry point of the program.
 *
 * Functional Utility: Handles multiple test cases. For each test case, it reads a word,
 *                     determines if it needs to be abbreviated based on its length
 *                     (greater than 10 characters), and prints either the abbreviated
 *                     form or the original word to standard output.
 *
 * @return 0 on successful execution.
 */
int main() {
  int n; // @brief Loop counter for iterating through test cases.
  int no_of_testcases; // @brief Stores the total number of test cases.
  cin >> no_of_testcases;
  // Block Logic: Iterate through each of the provided test cases.
  // Invariant: The loop continues as long as `n` is less than `no_of_testcases`.
  for (n = 0; n < no_of_testcases; n++) {
    string a; // @brief Stores the input word for the current test case.
    int length; // @brief Stores the calculated length of the input word.
    cin >> a;
    length = strlength(a);
    // Block Logic: Conditionally abbreviate the word based on its length.
    // Invariant: Words longer than 10 characters are abbreviated; otherwise, they are printed as is.
    if (length > 10)
      // Functional Utility: Prints the abbreviated word (first character, length-2, last character).
      cout << a[0] << length - 2 << a[length - 1] << endl;
    else
      // Functional Utility: Prints the original word as it is not long enough to be abbreviated.
      cout << a << endl;
  }
  return 0;
}
