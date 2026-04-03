/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces (Problem 71A).
 * @see https://codeforces.com/problemset/problem/71/A
 *
 * @description
 * This program processes a series of words. If a word is longer than 10 characters,
 * it is replaced with a special abbreviation. Otherwise, it is printed as is.
 * The abbreviation consists of the first character of the word, followed by the
 * number of characters between the first and last character, and finally the
 * last character of the word. For example, "localization" becomes "l10n".
 *
 * @algorithm
 * 1. Read the number of words, n.
 * 2. Loop n times to process each word.
 * 3. In each iteration, read the word.
 * 4. Check the length of the word.
 * 5. If the length is greater than 10, construct the abbreviation and print it.
 * 6. If the length is 10 or less, print the word as it is.
 *
 * @time_complexity O(N * L), where N is the number of words and L is the maximum
 *                  length of a word. This accounts for reading each word.
 * @space_complexity O(L) to store the longest word read from the input.
 */
#include <fstream>
#include <iostream>

using namespace std;

// A global flag to switch between standard input and file input for easy debugging.
bool DEBUG = false;

// A file stream object used when debugging.
ifstream fileStream;

/**
 * @brief Provides the appropriate input stream based on the DEBUG flag.
 * @return An istream reference to either standard input (cin) or a local file.
 *
 * @description
 * This is a helper function for competitive programming. When the DEBUG flag is true,
 * it redirects input from a local file named "input.txt", which is assumed to be
 * in the same directory as the source file. This allows for testing with predefined
 * inputs without manual entry.
 */
istream &getInputStream() {
  if (DEBUG) {
    string path = __FILE__;
    path = path.substr(0, 1 + path.find_last_of('/'));
    path += "input.txt";
    fileStream.open(path);
    return fileStream;
  }
  return cin;
}

/**
 * @brief Main logic to solve the problem for all test cases.
 */
void solve() {
  istream &inputStream = getInputStream();

  int n;
  inputStream >> n; // Read the number of words.

  string word;
  int wordLength;
  
  /**
   * @brief Block Logic: Process each word according to the abbreviation rule.
   * This loop iterates 'n' times, handling one word per iteration.
   */
  for (int i(0); i < n; i++) {
    inputStream >> word;
    wordLength = word.length();
    
    // Pre-condition: Check if the word is "too long".
    if (wordLength <= 10) {
      // If the word is not too long, print it as is.
      cout << word << endl;
      continue;
    }

    // If the word is too long, create and print the abbreviation.
    // The abbreviation is: [first letter][length-2][last letter].
    word = word[0] + to_string(wordLength - 2) + word[wordLength - 1];
    cout << word << endl;
  }
}

/**
 * @brief Main entry point of the program.
 * Sets up debugging if specified and calls the solve function.
 */
int main(int argc, char *argv[]) {
  /**
   * @brief Block Logic: Enable debug mode via command-line argument.
   * If the program is run with "debug" as the first argument, the DEBUG
   * flag is set to true, enabling file-based input.
   */
  if (argc >= 2) {
    string debug(argv[1]);
    if (debug == "debug") {
      DEBUG = true;
    }
  }
  
  solve();
  
  return 0;
}
