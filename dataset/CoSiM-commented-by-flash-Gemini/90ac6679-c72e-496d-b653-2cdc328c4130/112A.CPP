/**
 * @90ac6679-c72e-496d-b653-2cdc328c4130/112A.CPP
 * @brief Case-insensitive lexicographical string comparison simulation.
 * Domain: Competitive Programming, Algorithm Fundamentals.
 * Algorithm: Single-pass linear character sweep with normalization.
 * Time Complexity: $O(N)$ where N is the length of the input strings.
 * Space Complexity: $O(N)$ to store the entity sequences.
 */
#include <fstream>    // For file input stream operations (ifstream).
#include <iostream>   // For standard input/output operations (cin, cout).
#include <string>     // For string manipulation (std::string).
#include <cctype>     // For character case conversion (tolower).

using namespace std;

// Global flag to control debug mode behavior.
// When true, input is read from "input.txt"; otherwise, from standard input (cin).
bool DEBUG = false;

// Global ifstream object for reading from a file in debug mode.
ifstream fileStream;

/**
 * @brief Abstracted stream provider for environment-independent I/O.
 * Functional Utility: Decouples the core logic from the physical input source (stdin vs filesystem).
 *
 * @return An istream reference (either std::cin or a fileStream).
 */
istream &getInputStream() {
  if (DEBUG) {
    /**
     * Block Logic: Dynamically resolves the path to the local debug data file.
     * Strategy: Relative path derivation using the preprocessor __FILE__ macro.
     */
    string path = __FILE__;
    path = path.substr(0, 1 + path.find_last_of('/')); // Extracts directory path.
    path += "input.txt";                               // Appends "input.txt".
    fileStream.open(path);                             // Opens the input file.
    return fileStream;                                 // Returns the file stream.
  }
  return cin; // Returns standard input stream if not in debug mode.
}

/**
 * @brief Executes the string normalization and comparison logic.
 * Functional Utility: Implements the Petya and Strings normalization logic required by competitive constraints.
 */
void solve() {
  // Block Logic: Obtains the input stream, which could be cin or a file stream.
  istream &inputStream = getInputStream();

  string first, second;
  // Reads the two strings to be compared from the obtained input stream.
  inputStream >> first >> second;
  
  int32_t result = 0; // Stores the comparison result: 0 (equal), 1 (first > second), -1 (first < second).
  
  /**
   * Block Logic: Performs a normalized character-by-character comparison.
   * Invariant: 'result' remains 0 until a divergence in character value is detected after normalization.
   * Normalization: Maps all characters to lowercase to achieve case-insensitivity.
   */
  for (int i(0); i < first.length(); i++) {
    char a = tolower(first[i]);  // Normalizes character from 'first' string.
    char b = tolower(second[i]); // Normalizes character from 'second' string.
    
    // Logic: Identifies the first lexicographical divergence.
    if (a > b) {
      result = 1;  // 'first' string ranks higher.
      break;       // Optimization: Short-circuit once rank is determined.
    } else if (a < b) {
      result = -1; // 'first' string ranks lower.
      break;       // Optimization: Short-circuit once rank is determined.
    }
  }

  // Functional Utility: Emits the calculated rank to stdout.
  cout << result << endl;
}

/**
 * @brief Entry point for the comparison application.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * @return 0 for successful execution.
 */
int main(int argc, char *argv[]) {
  /**
   * Block Logic: Parses execution flags to determine the input mode.
   * Logic: Enables debug mode if the 'debug' flag is passed as an argument.
   */
  if (argc >= 2) {
    string debug(argv[1]);
    if (debug == "debug") {
      DEBUG = true; // Sets the global DEBUG flag to true.
    }
  }
  solve();    // Dispatches to the solver kernel.
  return 0; // Indicates successful program termination.
}
