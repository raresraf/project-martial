/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem from Codeforces (Problem 71A).
 * Problem Statement: Abbreviates words longer than 10 characters by replacing the middle
 *                    characters with the count of characters between the first and last.
 * Algorithm: Conditional abbreviation. Reads each word and checks its length. If longer
 *            than 10, it constructs an abbreviated string; otherwise, it keeps the original word.
 * Time Complexity: O(L) for each word, where L is the length of the word. Total O(N*L_avg)
 *                  where N is the number of words and L_avg is the average length.
 * Space Complexity: O(L) for storing each word.
 * Domain: Competitive Programming, String Manipulation.
 */
#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::to_string;

/**
 * @brief Main function to solve the "Way Too Long Words" problem.
 * Functional Utility: Processes a given number of words, applying abbreviation logic
 *                     to those exceeding a specific length and printing the result.
 */
int main(void) {
  // Variables:
  // lineCount: Number of test cases (words to process).
  // line: Input string (word).
  int lineCount;
  string line;

  // Block Logic: Reads the number of test cases.
  cin >> lineCount;
  // Block Logic: Processes each test case.
  // Invariant: In each iteration, a new word is read and processed according to the abbreviation rule.
  for (int i = 0; i < lineCount; i++) {
    // Block Logic: Reads the current word.
    cin >> line;
    size_t length = line.length();
    // Block Logic: Checks if the word's length is greater than 10 characters.
    // If true, abbreviates the word; otherwise, keeps the original word.
    if (length > 10) {
      // Functional Utility: Constructs the abbreviated word.
      // Logic: First character + (length - 2) as a string + last character.
      line = line[0] + to_string(length - 2) + line[length - 1];
    }
    // Functional Utility: Prints the processed word (original or abbreviated).
    cout << line << endl;
  }

  return 0;
}