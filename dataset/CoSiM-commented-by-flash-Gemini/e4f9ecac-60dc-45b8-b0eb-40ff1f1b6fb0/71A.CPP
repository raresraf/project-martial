/**
 * @file 71A.CPP
 * @brief This program solves the "Way Too Long Words" problem (Codeforces 71A).
 * It reads a series of words, and for each word, if its length exceeds 10
 * characters, it replaces the word with an abbreviation. The abbreviation
 * consists of the first character of the word, followed by the count of
 * characters between the first and last (as a number), and then the last
 * character. Words with 10 or fewer characters are printed as they are.
 * This version handles multiple test cases.
 *
 * Overall Design:
 * 1. Read the number of test cases.
 * 2. In a loop for each test case:
 *    a. Read an input word.
 *    b. Apply the abbreviation logic based on word length.
 *    c. Print the original or abbreviated word.
 *
 * Time Complexity: O(T * L), where T is the number of test cases and L is the
 * average length of a word. String operations contribute to this complexity.
 * Space Complexity: O(L) for storing the current input word.
 */
// CodeForces
// Problem 71A [Way Too Long Words]

#include <iostream> // Required for input/output operations (cin, cout)
#include <string>   // Required for string manipulation
using namespace std;

/**
 * @brief Main function where the program execution begins.
 *
 * This function manages the processing of multiple test cases for the
 * "Way Too Long Words" problem. It reads the number of test cases,
 * then iteratively processes each word according to the problem's rules.
 *
 * @return 0 if the program executes successfully.
 */
int main() {
  string input; // Declares a string variable to store each input word.

  unsigned int test_cases = 0; // Declares an unsigned integer to store the number of test cases.
  cin >> test_cases; // Reads the number of test cases from standard input.

  /**
   * @brief Loop to process each test case.
   *
   * This while loop decrements `test_cases` after each iteration,
   * continuing until all test cases have been processed.
   *
   * Pre-condition: `test_cases` holds the number of words to process.
   * Invariant: After each iteration, one word has been read, processed, and printed.
   */
  while (test_cases--) {
    cin >> input; // Reads the current word for the test case.

    /**
     * @brief Conditional block for word abbreviation.
     *
     * Checks if the length of the current `input` word is greater than 10.
     * - If true, it prints the abbreviated form (first character, length-2, last character).
     * - If false, it prints the original word.
     */
    if (input.length() > 10) {
      cout << input[0]                           // Prints the first character of the word.
           << input.length() - 2                 // Prints the count of characters between the first and last.
           << input[input.length() - 1]          // Prints the last character of the word.
           << endl; // Newline for the next output.
    } else {
      cout << input << endl; // Prints the original word followed by a newline.
    }
  }
  return 0; // Indicates successful program execution.
}