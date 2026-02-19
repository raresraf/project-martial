/**
 * @file 236A.CPP
 * @brief Solution to a competitive programming problem determining user interaction based on username character diversity.
 *
 * This program analyzes a given username string to determine whether to "CHAT WITH HER!" or "IGNORE HIM!".
 * The decision is based on the parity of the number of unique characters in the username.
 *
 * @algorithm The core logic uses a std::set to efficiently count the number of unique characters in the input string.
 *            The final determination is made by checking if the size of the set is even or odd.
 *
 * @time_complexity O(N), where N is the length of the input string. Each character insertion into the std::set takes O(log K) time,
 *                  where K is the number of unique characters. Since K is bounded by the size of the character set (a small constant),
 *                  the effective complexity is linear in the length of the string.
 *
 * @space_complexity O(K), where K is the number of unique characters. This is effectively O(1) as the size is bounded by the
 *                   character set size.
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>

using namespace std;

bool DEBUG = false;

ifstream fileStream;

/**
 * @brief Gets the input stream, switching between standard input and a file for debugging.
 *
 * @return A reference to the appropriate istream. If DEBUG is true, it returns a file stream
 *         to "input.txt". Otherwise, it returns standard input (cin).
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
 * @brief Main logic to solve the problem.
 *
 * Reads a username, counts its unique characters, and prints the result based on the count's parity.
 */
void solve() {
  istream &inputStream = getInputStream();

  // Your code goes here
  string name;
  inputStream >> name;
  set<char> uniqueChars;
  /**
   * Block Logic: Iterate through the username to identify all unique characters.
   * Invariant: `uniqueChars` contains all unique characters from the substring `name[0...i-1]`.
   */
  for (int i(0); i < name.length(); i++) {
    uniqueChars.insert(name[i]);
  }

  /**
   * Block Logic: Determine the output based on the parity of the number of unique characters.
   * Pre-condition: `uniqueChars` holds all unique characters from the input string `name`.
   */
  cout << (((uniqueChars.size() & 1) == 1) ? "IGNORE HIM!" : "CHAT WITH HER!");
}

/**
 * @brief Main function, entry point of the program.
 *
 * Handles command-line arguments to enable debug mode and then calls the solver function.
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return 0 on successful execution.
 */
int main(int argc, char *argv[]) {
  /**
   * Block Logic: Check for a "debug" command-line argument to switch input to a local file.
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
