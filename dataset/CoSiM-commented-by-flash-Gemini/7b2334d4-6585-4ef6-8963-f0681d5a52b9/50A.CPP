/**
 * @file 50A.CPP
 * @brief Solution to a competitive programming problem: Domino piling.
 *
 * Algorithm: Direct calculation based on board dimensions.
 * Time Complexity: O(1).
 * Space Complexity: O(1).
 *
 * This program calculates the maximum number of 1x2 dominoes that can be placed
 * on an M x N rectangular board. Each domino covers exactly two squares.
 * The solution is simply half the total number of squares (M * N / 2 using integer division).
 */

#include <fstream>
#include <iostream>

// Functional Utility: Simplifies standard namespace access for competitive programming.
using namespace std;

// Global flag to enable or disable debug mode, affecting input source.
bool DEBUG = false;

ifstream fileStream;

/**
 * @brief Provides an input stream based on the DEBUG flag.
 *
 * @return std::istream& Reference to std::cin for normal mode, or to a fileStream for debug mode.
 */
istream &getInputStream() {
  // Block Logic: Determines whether to use standard input (cin) or file input for debugging.
  // Invariant: An appropriate input stream is returned based on the DEBUG flag.
  if (DEBUG) {
    // Functional Utility: Constructs the path to 'input.txt' relative to the source file.
    string path = __FILE__;
    path = path.substr(0, 1 + path.find_last_of('/'));
    path += "input.txt";
    // Functional Utility: Opens 'input.txt' for reading when in debug mode.
    fileStream.open(path);
    return fileStream;
  }
  return cin;
}

/**
 * @brief Solves the domino piling problem.
 *
 * Reads board dimensions M and N, then prints the maximum number of dominoes.
 */
void solve() {
  // Functional Utility: Obtains the appropriate input stream.
  istream &inputStream = getInputStream();

  int m, n;
  // Functional Utility: Reads the two dimensions of the rectangular board.
  inputStream >> m >> n;

  // Functional Utility: Calculates and prints the maximum number of dominoes using integer division.
  cout << (m * n / 2) << endl;
}

/**
 * @brief Main function of the program.
 *
 * Handles debug mode activation and calls the solve function.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * @return int Exit status of the program.
 */
int main(int argc, char *argv[]) {
  // Block Logic: Checks for a command-line argument to enable debug mode.
  // Invariant: The DEBUG flag is set if "debug" is provided as the first argument.
  if (argc >= 2) {
    string debug(argv[1]);
    if (debug == "debug") {
      DEBUG = true;
    }
  }
  // Functional Utility: Invokes the problem-solving logic.
  solve();
  return 0;
}