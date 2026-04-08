/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem from Codeforces (Problem 71A).
 * Problem Statement: Abbreviates words longer than 10 characters by replacing the middle
 *                    characters with the count of characters between the first and last.
 * Algorithm: Conditional abbreviation. Reads each word and checks its length. If longer
 *            than 10, it constructs an abbreviated string; otherwise, it prints the original word.
 * Time Complexity: O(L) for each word, where L is the length of the word. Total O(N*L_avg)
 *                  where N is the number of words and L_avg is the average length.
 * Space Complexity: O(L) for storing each word.
 * Domain: Competitive Programming, String Manipulation.
 */
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Main function to solve the "Way Too Long Words" problem.
 * Functional Utility: Processes a given number of words, applying abbreviation logic
 *                     to those exceeding a specific length and printing the result.
 */
int main() {
  // Variables:
  // t: Number of test cases (words to process).
  // str: Input string (word).
  int t;
  string str;
  // Block Logic: Reads the number of test cases.
  cin >> t;
  // Block Logic: Processes each test case.
  // Invariant: In each iteration, a new word is read and processed according to the abbreviation rule.
  while (t--) {
    // Block Logic: Reads the current word.
    cin >> str;
    // Block Logic: Checks if the word's length is greater than 10 characters.
    // If true, abbreviates the word; otherwise, prints the original word.
    if (str.length() > 10)
      // Functional Utility: Constructs and prints the abbreviated word.
      // Logic: First character + (length - 2) as a string + last character.
      cout << str[0] << str.length() - 2 << str[str.length() - 1] << endl;
    else
      // Functional Utility: Prints the original word as it is not long enough to be abbreviated.
      cout << str << endl;
  }
  return 0;
}