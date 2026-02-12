/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem.
 * @details This program reads a single word from the standard input (or a file if in debug mode)
 * and capitalizes its first letter. The rest of the word remains unchanged. The modified word
 * is then printed to the standard output. This is a common introductory problem in competitive programming
 * that tests basic string manipulation.
 *
 * @algorithm String modification.
 * @complexity
 * Time complexity: O(L) for reading the string of length L. The capitalization is O(1).
 * Space complexity: O(L) to store the word.
 */
#include <algorithm>
#include <fstream>
#include <iostream>

using namespace std;

// DEBUG: A flag to switch between standard input and file input for debugging purposes.
bool DEBUG = false;

ifstream fileStream;

/**
 * @brief Gets the input stream, either from a file or standard input.
 * @return An istream reference, which is `cin` by default or a file stream if `DEBUG` is true.
 * @details If `DEBUG` is true, it constructs a path to "input.txt" relative to the source file's location,
 * opens it, and returns the file stream. Otherwise, it returns `cin`.
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
 * @brief The core logic of the solution.
 */
void solve() {
  istream &inputStream = getInputStream();

  // Your code goes here
  string word;
  inputStream >> word;

  // Capitalize the first letter of the word.
  word[0] = toupper(word[0]);
  cout << word << endl;
}

/**
 * @brief Main function.
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return 0 on successful execution.
 * @details It checks for a "debug" command-line argument to set the `DEBUG` flag, then calls `solve()`.
 */
int main(int argc, char *argv[]) {
  // Check for "debug" argument to enable file-based input.
  if (argc >= 2) {
    string debug(argv[1]);
    if (debug == "debug") {
      DEBUG = true;
    }
  }
  solve();
  return 0;
}