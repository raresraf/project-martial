/**
 * @file 281A.CPP
 * @brief Solution for Codeforces Problem 281A - Word Capitalization.
 *
 * This problem asks to capitalize the first letter of a given word (string) and print the result.
 * This solution includes a common competitive programming debugging setup that allows
 * input to be read from a file (`input.txt`) when a "debug" argument is provided.
 *
 * Algorithm: Character manipulation.
 * The core logic reads a single word (string), converts its first character to
 * uppercase using `toupper()`, and then prints the modified string.
 *
 * Time Complexity: O(L), where L is the length of the input string, due to string
 * input/output and a constant number of operations on the first character.
 * Space Complexity: O(L) for storing the input string.
 */

#include <algorithm> // For general-purpose algorithms like `std::transform`, though `toupper` is directly used here.
#include <fstream> // For file stream operations (`ifstream`), used for debug input.
#include <iostream> // For standard C++ input/output operations (`cin`, `cout`).
#include <string> // For using the `std::string` class.
#include <cctype> // For character manipulation functions like `toupper`.

using namespace std; // Functional Utility: Brings all identifiers from the std namespace into the current scope.

bool DEBUG = false; // Global flag to enable/disable debug mode (reading from file).

ifstream fileStream; // Global file stream object for reading input from a file in debug mode.

/**
 * @brief Provides an input stream, either from `cin` or an "input.txt" file based on `DEBUG` flag.
 * Functional Utility: Abstracts the source of input, allowing for easy switching between
 * standard input and a file for testing and debugging purposes.
 * @return A reference to an `istream` object.
 */
istream &getInputStream() {
  if (DEBUG) { // Block Logic: If debug mode is enabled.
    string path = __FILE__; // Get the path of the current source file.
    path = path.substr(0, 1 + path.find_last_of('/')); // Extract the directory path.
    path += "input.txt"; // Append the input file name.
    fileStream.open(path); // Open the input file.
    return fileStream; // Return the file stream.
  }
  return cin; // If not in debug mode, return standard input.
}

/**
 * @brief Contains the core logic to solve the Word Capitalization problem.
 * Functional Utility: Reads a word, capitalizes its first letter, and prints the result.
 * It uses `getInputStream()` to get its input source.
 */
void solve() {
  istream &inputStream = getInputStream(); // Get the appropriate input stream.

  // Your code goes here
  string word; // Declares a string variable `word` to store the input.
  inputStream >> word; // Reads a word from the input stream.

  // Functional Utility: Capitalizes the first character of the word.
  // Precondition: `word` is not an empty string.
  // Postcondition: If `word[0]` is a lowercase letter, it becomes uppercase; otherwise, it remains unchanged.
  word[0] = toupper(word[0]);
  cout << word << endl; // Prints the modified word followed by a newline.
}

/**
 * @brief Main entry point of the program.
 * Functional Utility: Parses command-line arguments to determine debug mode,
 * then calls the `solve()` function to execute the problem logic.
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * @return 0 on successful execution.
 */
int main(int argc, char *argv[]) {
  if (argc >= 2) { // Block Logic: Check if a command-line argument is provided.
    string debug(argv[1]); // Convert the first argument to a string.
    if (debug == "debug") { // Check if the argument is "debug".
      DEBUG = true; // Set the global DEBUG flag to true.
    }
  }
  solve(); // Call the function containing the problem's solution.
  return 0; // Indicate successful program execution.
}