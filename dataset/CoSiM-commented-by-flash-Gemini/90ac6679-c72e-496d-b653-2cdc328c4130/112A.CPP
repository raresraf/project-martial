/**
 * @file 112A.CPP
 * @brief Lexicographically compares two strings after converting all characters to lowercase.
 *        Supports a debug mode to read input from a file named 'input.txt'.
 * Algorithm: Character-by-character comparison of two strings, converting each character
 *            to lowercase before comparison. The comparison stops at the first differing
 *            character or when the end of the shorter string is reached.
 * Time Complexity: O(L) where L is the length of the shorter string, as characters are
 *                  processed linearly until a difference is found or an end is reached.
 * Space Complexity: O(L) for storing the input strings.
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
 * @brief Returns an appropriate input stream based on the DEBUG flag.
 * Functional Utility: Abstracts the source of input, allowing easy switching
 *                     between standard input and file input for debugging.
 *
 * @return An istream reference (either std::cin or a fileStream).
 */
istream &getInputStream() {
  if (DEBUG) {
    // Block Logic: Constructs the path to "input.txt" relative to the source file.
    string path = __FILE__;
    path = path.substr(0, 1 + path.find_last_of('/')); // Extracts directory path.
    path += "input.txt";                               // Appends "input.txt".
    fileStream.open(path);                             // Opens the input file.
    return fileStream;                                 // Returns the file stream.
  }
  return cin; // Returns standard input stream if not in debug mode.
}

/**
 * @brief Solves the string comparison problem.
 * Functional Utility: Reads two strings, performs a case-insensitive lexicographical
 *                     comparison, and prints 1, -1, or 0 based on the result.
 */
void solve() {
  // Block Logic: Obtains the input stream, which could be cin or a file stream.
  istream &inputStream = getInputStream();

  string first, second;
  // Reads the two strings to be compared from the obtained input stream.
  inputStream >> first >> second;
  
  int32_t result = 0; // Stores the comparison result: 0 (equal), 1 (first > second), -1 (first < second).
  
  // Block Logic: Iterates through the strings character by character up to the length of the shorter string.
  // Invariant: 'result' is 0 until a difference is found, at which point it's set and the loop breaks.
  for (int i(0); i < first.length(); i++) {
    char a = tolower(first[i]);  // Converts character from 'first' string to lowercase.
    char b = tolower(second[i]); // Converts character from 'second' string to lowercase.
    
    // Compares the lowercase characters.
    if (a > b) {
      result = 1;  // 'first' is lexicographically greater.
      break;       // No need to compare further.
    } else if (a < b) {
      result = -1; // 'first' is lexicographically smaller.
      break;       // No need to compare further.
    }
    // If a == b, loop continues to the next character.
  }

  // Functional Utility: Prints the comparison result to standard output.
  cout << result << endl;
}

/**
 * @brief Entry point of the program.
 * Functional Utility: Initializes debug mode if specified via command-line arguments,
 *                     then calls the solve function.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * @return 0 for successful execution.
 */
int main(int argc, char *argv[]) {
  // Block Logic: Checks if command-line arguments are provided and if the first argument is "debug".
  if (argc >= 2) {
    string debug(argv[1]);
    if (debug == "debug") {
      DEBUG = true; // Sets the global DEBUG flag to true.
    }
  }
  solve();    // Calls the function to execute the string comparison logic.
  return 0; // Indicates successful program termination.
}
