/**
 * @498724e6-f1a6-4a4c-9190-d7242b862b0e/282A.CPP
 * @brief This program simulates a simple variable manipulation process,
 *        incrementing or decrementing an integer variable 'x' based on input statements.
 * Algorithm: Iterative processing of input strings, checking for "++" or "--" patterns.
 * Time Complexity: O(N * L) where N is the number of statements and L is the average length of a statement.
 * Space Complexity: O(L) for storing the input line.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  int x = 0; // Initialize variable 'x' to 0. This variable will be incremented or decremented.

  int n; // Declare 'n' to store the number of statements.
  // Functional Utility: Read the number of statements.
  cin >> n;

  string line; // Declare a string 'line' to store each input statement.
  // Block Logic: Iterate 'n' times, processing each statement.
  for (int i = 0; i < n; i++) {
    // Functional Utility: Read the current statement.
    cin >> line;
    // Pre-condition: Check if the statement contains "++" for increment operation.
    if (line.find("++") != string::npos) {
      x++; // Action: Increment 'x'.
    } else {
      x--; // Action: Otherwise (implied "--"), decrement 'x'.
    }
  }

  // Functional Utility: Output the final value of 'x'.
  cout << x << endl;
}
