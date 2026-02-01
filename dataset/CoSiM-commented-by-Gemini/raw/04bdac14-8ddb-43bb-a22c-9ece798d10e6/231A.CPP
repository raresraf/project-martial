/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces contest 231.
 *
 * @details
 * This program calculates the number of problems a three-person team will implement.
 * The team decides to solve a problem only if at least two members are confident in the solution.
 * The input provides the number of problems and, for each problem, three boolean values (0 or 1)
 * indicating each member's confidence.
 *
 * The program features a debug mode, which can be activated by passing "debug" as a command-line
 * argument. In debug mode, input is read from a file named "input.txt" located in the same
 * directory as the source file, instead of from standard input.
 *
 * Algorithm:
 * 1. Check for a "debug" command-line argument to switch the input stream to a file if needed.
 * 2. Read the total number of problems, `n`.
 * 3. Initialize a counter for the problems to be implemented to zero.
 * 4. Loop `n` times, once for each problem:
 *    a. Initialize a temporary counter for the current problem to zero.
 *    b. Read the three integer opinions (0 or 1).
 *    c. Sum the three opinions.
 *    d. If the sum is 2 or greater, increment the main problem counter.
 * 5. Print the final count of problems to be implemented.
 *
 * Time Complexity: O(n), where `n` is the number of problems, due to the single loop through all problems.
 * Space Complexity: O(1), as the program uses a constant amount of memory regardless of the input size.
 */
#include <fstream>
#include <iostream>

using namespace std;

// Global flag to control debug mode.
bool DEBUG = false;

// File stream for reading input from a file in debug mode.
ifstream fileStream;

/**
 * @brief Returns the appropriate input stream based on the DEBUG flag.
 * @return A reference to `cin` for standard input, or a reference to a file stream if in debug mode.
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
 * @brief Contains the core logic for solving the problem.
 */
void solve() {
  istream &inputStream = getInputStream();

  int n;
  // Read the number of problems.
  inputStream >> n;
  int numProblems = 0; // Counter for the number of problems to be implemented.
  
  // Block Logic: Iterate through each of the 'n' problems.
  for (int i(0); i < n; i++) {
    int count = 0; // Counter for members sure about the current problem.
    // Block Logic: Read the opinions of the three team members.
    for (int j(0); j < 3; j++) {
      int solved;
      inputStream >> solved;
      count += solved; // Sum the opinions (1 for sure, 0 for not sure).
    }
    // Pre-condition: Check if at least two members are sure.
    if (count >= 2) {
      numProblems++; // Increment the count of problems to implement.
    }
  }
  cout << numProblems;
}

int main(int argc, char *argv[]) {
  // Command-line argument parsing to enable debug mode.
  if (argc >= 2) {
    string debug(argv[1]);
    if (debug == "debug") {
      DEBUG = true;
    }
  }
  solve();
  return 0;
}