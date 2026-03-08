/**
 * @file 112A.CPP
 * @brief A well-structured solution for case-insensitive string comparison (e.g., Codeforces 112A).
 * @details This program compares two strings lexicographically, ignoring case. It features
 * a clean abstraction for switching between standard input and a local file (`input.txt`)
 * for easier debugging.
 *
 * @algorithm Character-by-character, case-insensitive comparison.
 * @complexity
 * - Time: O(L), where L is the length of the shorter string, as the loop terminates
 *   at the first point of difference.
 * - Space: O(L) to store the two input strings.
 */

#include <fstream>
#include <iostream>
#include <string> // It's good practice to include necessary headers explicitly.
#include <cctype> // For tolower()

using namespace std;

// A flag to switch between standard input and file input for debugging.
bool DEBUG = false;

// A global filestream object used when debugging.
ifstream fileStream;

/**
 * @brief Returns an input stream, either `cin` or a file stream if in debug mode.
 * @return A reference to the appropriate `istream` object.
 */
istream &getInputStream() {
  if (DEBUG) {
    // Construct the path to "input.txt" relative to this source file.
    string path = __FILE__;
    path = path.substr(0, 1 + path.find_last_of('/'));
    path += "input.txt";
    fileStream.open(path);
    return fileStream;
  }
  return cin;
}

/**
 * @brief Contains the core logic for solving the problem.
 */
void solve() {
  istream &inputStream = getInputStream();

  string first, second;
  inputStream >> first >> second;
  int32_t result = 0; // Default result is 0 (equal).

  /**
   * @block Lexicographical Comparison
   * @logic Iterates through the strings, converting characters to lowercase on-the-fly
   * and comparing them. The loop breaks as soon as a difference is found.
   */
  for (int i = 0; i < first.length(); i++) {
    char a = tolower(first[i]);
    char b = tolower(second[i]);
    if (a > b) {
      result = 1;
      break;
    } else if (a < b) {
      result = -1;
      break;
    }
  }

  cout << result << endl;
}

int main(int argc, char *argv[]) {
  // Enable debug mode if "debug" is passed as a command-line argument.
  if (argc >= 2) {
    string debug(argv[1]);
    if (debug == "debug") {
      DEBUG = true;
    }
  }
  solve();
  return 0;
}
