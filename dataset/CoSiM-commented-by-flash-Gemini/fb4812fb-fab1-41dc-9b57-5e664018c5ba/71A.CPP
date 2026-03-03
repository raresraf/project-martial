/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A - "Way Too Long Words".
 * This program abbreviates words that are strictly longer than 10 characters.
 * An abbreviation consists of the first letter, the count of characters between
 * the first and last letters, and the last letter. Words 10 characters or shorter
 * remain unchanged.
 * Algorithm: String length check and conditional abbreviation.
 * Time Complexity: O(N * L), where N is the number of words and L is the average length of a word.
 *                  Each word is processed in time proportional to its length.
 * Space Complexity: O(L) - to store the longest word.
 */
#include <fstream>    // Required for file input/output operations (ifstream).
#include <iostream>   // Required for standard input/output operations (cin, cout).
#include <string>     // Required for string manipulation.

using namespace std; // Uses the standard namespace.

// Global flag to control debug mode, enabling input from a file.
bool DEBUG = false;

// ifstream object used for reading from a file when in debug mode.
ifstream fileStream;

/**
 * @brief Provides an input stream based on the DEBUG flag.
 * If DEBUG is true, it opens "input.txt" and returns an ifstream.
 * Otherwise, it returns the standard input stream (cin).
 * Functional Utility: Allows easy switching between sample input from a file and standard console input.
 *
 * @return An istream reference for reading input.
 */
istream &getInputStream() {
  if (DEBUG) {
    // Construct the path to "input.txt" relative to the current file.
    string path = __FILE__;
    path = path.substr(0, 1 + path.find_last_of('/')); // Extracts directory path.
    path += "input.txt"; // Appends the input file name.
    fileStream.open(path); // Open the input file.
    return fileStream;     // Return the file input stream.
  }
  return cin; // Return standard input stream.
}

/**
 * @brief Solves the "Way Too Long Words" problem.
 * Reads words and prints them, abbreviating those longer than 10 characters.
 */
void solve() {
  istream &inputStream = getInputStream(); // Get the appropriate input stream.

  int n;            // Declares an integer to store the number of words.
  inputStream >> n; // Reads the number of words.

  string word;      // Declares a string to store each word.
  int wordLength;   // Declares an integer to store the length of the current word.

  /**
   * Block Logic: Processes each word according to the problem's abbreviation rules.
   * Invariant: After each iteration, one word (original or abbreviated) is printed.
   */
  for (int i(0); i < n; i++) { // Loop 'n' times, once for each word.
    inputStream >> word;      // Reads a word from the input stream.
    wordLength = word.length(); // Gets the length of the current word.

    // Checks if the word is "too long" (length strictly greater than 10).
    if (wordLength <= 10) {
      cout << word << endl; // If not too long, print the word as is.
      continue;             // Move to the next word.
    }

    // If the word is too long, construct its abbreviation.
    // Abbreviation: first character + (length - 2) as string + last character.
    word = word[0] + to_string(wordLength - 2) + word[wordLength - 1];
    cout << word << endl; // Print the abbreviated word.
  }
}

/**
 * @brief Main function of the program.
 * Handles command-line arguments for debug mode and calls the solve function.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * @return 0 if execution is successful.
 */
int main(int argc, char *argv[]) {
  // Checks if a command-line argument is provided (e.g., "debug").
  if (argc >= 2) {
    string debug(argv[1]); // Convert the argument to a string.
    if (debug == "debug") {
      DEBUG = true; // Set the global DEBUG flag to true.
    }
  }
  solve(); // Call the function that implements the problem's solution.
  return 0; // Indicates successful program execution.
}