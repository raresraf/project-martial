/**
 * @file 71A.CPP
 * @brief Abbreviates words that are too long.
 *
 * This program solves the "Way Too Long Words" problem. It reads a series of
 * words and for any word longer than 10 characters, it prints an abbreviation:
 * the first character, the number of characters in between, and the last character.
 * Shorter words are printed unchanged. This version uses a custom function to
 * calculate string length.
 *
 * Algorithm:
 * 1. Read the number of test cases.
 * 2. Loop for each test case:
 *    a. Read the word.
 *    b. Calculate its length.
 *    c. If the length is greater than 10, print the abbreviation.
 *    d. Otherwise, print the original word.
 *
 * Time Complexity: O(T * L), where T is the number of test cases and L is the
 * maximum length of a word. The custom length function adds to the constant
 * factor for each word but doesn't change the overall complexity.
 * Space Complexity: O(L) to store the longest word.
 */
#include <iostream>
#include <string>
using namespace std;

/**
 * @brief Calculates the length of a C-style string terminated by a null character.
 * @param a The input std::string.
 * @return The number of characters before the null terminator.
 * @note This is a manual implementation and is less efficient than the built-in
 *       `a.length()` or `a.size()` methods provided by `std::string`. It treats
 *       `std::string` like a C-style char array.
 */
int strlength(string a) {
  int count = 0;
  // Invariant: count is the number of characters processed so far.
  while (a[count] != '\0') {
    count++;
  }
  return count;
}

int main() {
  int n, no_of_testcases;
  // Read the number of words to be processed.
  cin >> no_of_testcases;
  
  /**
   * Block Logic: Main processing loop.
   * Iterates through each test case.
   */
  for (n = 0; n < no_of_testcases; n++) {
    string a;
    int length;
    cin >> a;
    
    // Calculate the length using the custom function.
    length = strlength(a);

    /**
     * Block Logic: Abbreviation rule.
     * Pre-condition: A word `a` and its `length` have been determined.
     * If the word's length is greater than 10, it is abbreviated.
     * Otherwise, it is printed as is.
     */
    if (length > 10)
      cout << a[0] << length - 2 << a[length - 1] << endl;
    else
      cout << a << endl;
  }
  return 0;
}
