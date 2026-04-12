/**
 * @file 71A.CPP
 * @brief This program implements a solution to the "Way Too Long Words" problem
 * from competitive programming (e.g., Codeforces 71A).
 * The task is to read a list of words and, for each word longer than 10 characters,
 * replace it with an abbreviation. The abbreviation consists of the first
 * character, the count of characters between the first and last (as a number),
 * and the last character. Words 10 characters or shorter are printed as is.
 *
 * Overall Design:
 * 1. Read the number of words 'w'.
 * 2. Loop 'w' times to read each word into a vector of strings.
 * 3. Iterate through the vector, applying the abbreviation logic or printing the
 *    original word based on its length.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is the
 * average length of a word. String operations like `length()`, `at()`, and
 * `substr()` take time proportional to their length.
 * Space Complexity: O(N * L) for storing all the words in the vector.
 */
#include <iostream> // Required for input/output operations (cin, cout)
#include <string>   // Required for string manipulation
#include <vector>   // Required for using std::vector to store strings
using namespace std;

/**
 * @brief Main function where the program execution begins.
 *
 * This function orchestrates reading words, applying the abbreviation logic,
 * and printing the results as per the problem statement.
 *
 * @return 0 if the program executes successfully.
 */
int main() {
  int w{0}; // Declare an integer 'w' and initialize to 0. This will store the
            // number of words to be processed.
  vector<string> v; // Declare a vector of strings 'v' to store the input words.
  cin >> w;         // Read the number of words 'w' from standard input.

  /**
   * @brief Loop to read input words.
   *
   * This loop iterates 'w' times, reading each word from standard input
   * and storing it into the vector 'v'.
   *
   * Pre-condition: 'w' contains the total count of words to read.
   * Invariant: After each iteration, 'v' contains 'i+1' words.
   */
  for (int i{0}; i < w; i++) {
    string s;  // Declare a temporary string 's' to hold the current word.
    cin >> s;  // Read a word from standard input.
    v.push_back(s); // Add the read word to the end of the vector 'v'.
  }

  /**
   * @brief Loop to process and print words with abbreviation logic.
   *
   * This loop iterates through each string (word) stored in the vector 'v'.
   * It checks the length of each word:
   * - If the length is greater than 10, it constructs and prints an abbreviation.
   * - Otherwise, it prints the original word.
   *
   * Pre-condition: 'v' contains all input words.
   * Invariant: Words up to the current iteration 's' have been processed and printed.
   */
  for (auto s : v) { // Iterate over each string 's' in the vector 'v'.
    if (s.length() > 10) { // Check if the word's length is greater than 10.
      // If true, print the abbreviated form:
      // First character + (length - 2) as a number + last character.
      // E.g., "localization" becomes "l10n".
      cout << s.at(0) // Print the first character.
           << s.substr(1, s.length() - 2).length() // Calculate and print the count of middle characters.
           << s.at(s.length() - 1) // Print the last character.
           << endl; // Newline for the next output.
    } else { // If the word's length is 10 or less.
      cout << s << endl; // Print the original word followed by a newline.
    }
  }
  return 0; // Indicate successful execution.
}
