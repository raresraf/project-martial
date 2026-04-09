/**
 * @file 71A.CPP
 * @brief This program implements a word abbreviation logic.
 * Words longer than 10 characters are abbreviated to their first letter,
 * followed by the count of characters between the first and last letters,
 * and then the last letter. Words 10 characters or shorter remain unchanged.
 * This is a common string manipulation problem in competitive programming.
 *
 * Algorithm: String length check and formatting.
 * 1. Read the number of words to process.
 * 2. For each word:
 *    a. Read the word.
 *    b. Check its length.
 *    c. If the length is greater than 10, construct and print the abbreviated form.
 *       The length of the middle part is calculated using `s.substr(1, s.length() - 2).length()`,
 *       which simplifies to `s.length() - 2`.
 *    d. Otherwise, print the word as is.
 * Time Complexity: O(Sum of all word lengths) where each character is processed at most once.
 * Space Complexity: O(Max_Word_Length) to store the current word.
 */

// solutions.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started:
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see buil4d output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add
//   Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project
//   and select the .sln file
#include <iostream>
#include <string> // Explicitly include string header for clarity.
using namespace std;

/**
 * @brief Main function of the program.
 * Reads words, applies the abbreviation logic if necessary, and prints the result.
 * @pre Input consists of an integer `amt` (number of words), followed by `amt` word strings.
 * @post For each word, prints its abbreviated form (if applicable) to standard output, each on a new line.
 * @return 0 upon successful execution.
 */
int main() {
  // Functional Utility: `amt` stores the total number of words to process.
  int amt;
  // Functional Utility: Reads the total number of words `amt`.
  cin >> amt;
  // Block Logic: Iterates `amt` times to process each word.
  // Invariant: At each iteration, a new word is read and processed according to the abbreviation rules.
  for (int i = 0; i < amt; i++) {
    // Functional Utility: `s` stores the current word being processed.
    string s;
    // Functional Utility: Reads the current word.
    cin >> s;
    // Block Logic: Checks if the word's length exceeds 10 characters to determine if abbreviation is needed.
    // Pre-condition: `s` contains the current word.
    if (s.length() <= 10) {
      cout << s << endl; // Functional Utility: Prints the word as is if it's 10 characters or shorter.
    } else {
      // Functional Utility: Prints the abbreviated form: first character + (length - 2) + last character.
      // `s.substr(1, s.length() - 2).length()` effectively calculates `s.length() - 2`.
      cout << s[0] << s.substr(1, s.length() - 2).length() << s[s.length() - 1]
           << endl;
    }
  }
  return 0;
}
