/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem from Codeforces.
 * @details This program calculates the maximum number of 2x1 dominoes that can be
 * placed on a rectangular board of size M x N.
 *
 * @see https://codeforces.com/problemset/problem/50/A
 *
 * Algorithm:
 * The total area of the board is M * N. Each domino covers an area of 2.
 * Therefore, the maximum number of dominoes that can fit is the total area
 * divided by the area of a single domino. Since we are dealing with integers,
 * this is equivalent to integer division `(M * N) / 2`.
 *
 * Time Complexity: O(1), as it involves a few arithmetic operations.
 * Space Complexity: O(1).
 *
 * The code also includes a debugging feature to read input from a local file
 * `input.txt` instead of standard input when the program is run with the
 * "debug" command-line argument.
 */

#include <fstream>
#include <iostream>

using namespace std;

// A global flag to enable or disable debug mode.
bool DEBUG = false;

// A file stream object to be used when reading from a file in debug mode.
ifstream fileStream;

/**
 * @brief Returns an input stream, either `cin` or a file stream for debugging.
 * @return A reference to the appropriate `istream` object.
 */
istream &getInputStream() {
  if (DEBUG) {
    // Construct the path to "input.txt" relative to the source file.
    string path = __FILE__;
    path = path.substr(0, 1 + path.find_last_of('/'));
    path += "input.txt";
    fileStream.open(path);
    return fileStream;
  }
  // Default to standard input.
  return cin;
}

/**
 * @brief The core logic for solving the problem.
 */
void solve() {
  istream &inputStream = getInputStream();

  int m, n;
  inputStream >> m >> n;

  // The number of dominoes is the total area divided by 2.
  cout << (m * n / 2) << endl;
}

/**
 * @brief The main function.
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line arguments.
 * @return 0 on successful execution.
 */
int main(int argc, char *argv[]) {
  // Check for the "debug" command-line argument to enable debug mode.
  if (argc >= 2) {
    string debug(argv[1]);
    if (debug == "debug") {
      DEBUG = true;
    }
  }
  solve();
  return 0;
}
