/**
 * @file 58A.CPP
 * @brief Solution to a competitive programming problem that checks if the word "hello"
 * can be formed as a subsequence from a given input string.
 * A subsequence is a sequence that can be derived from another sequence by deleting
 * some or no elements without changing the order of the remaining elements.
 * Algorithm: Linear scan with state-tracking to match characters of "hello" sequentially.
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(L) for storing the input string.
 */
#include <iostream> // For input/output operations.
#include <string>   // For string manipulation.

using namespace std;

int main() {
  string input; // Declares 'input' to store the input string.
  // Reads an entire line from standard input into 'input'.
  getline(cin, input);

  int n; // Declares 'n' to store the length of the input string.
  n = input.length(); // Gets the length of the input string.

  // 'flag' acts as a state machine to track the progress of matching "hello".
  // 0: looking for 'h', 1: looking for 'e', 2: looking for first 'l', 3: looking for second 'l', 4: looking for 'o', 5: "hello" found.
  int flag = 0;

  // cout<<n<<'\n'; // Commented out debug statement.

  /**
   * Block Logic: Iterates through each character of the input string.
   * Invariant: At the start of each iteration, 'i' is the current character's index,
   * and 'flag' indicates how much of "hello" has been matched as a subsequence so far.
   */
  for (int i = 0; i < n; i++) {
    /**
     * Conditional Logic: Checks if the current character matches the next expected character
     * in the "hello" sequence, based on the current 'flag' state.
     */
    if ((flag == 0 && input.at(i) == 'h') ||          // Matches 'h' if `flag` is 0.
        (flag == 1 && input.at(i) == 'e') ||          // Matches 'e' if `flag` is 1.
        ((flag == 2 || flag == 3) && input.at(i) == 'l') || // Matches 'l' if `flag` is 2 or 3.
        (flag == 4 && input.at(i) == 'o')) {          // Matches 'o' if `flag` is 4.
      flag++; // Increments 'flag' to move to the next state in matching "hello".
    }
    // Conditional Logic: If 'flag' reaches 5, it means "hello" has been fully matched.
    if (flag == 5) {
      cout << "YES"; // Prints "YES".
      return 0;      // Terminates the program early as the solution is found.
    }
    // cout<<i; // Commented out debug statement.
  }

  // cout<<flag; // Commented out debug statement.

  // If the loop completes without 'flag' reaching 5, "hello" was not found.
  cout << "NO";
  return 0; // Indicates successful program execution.
}
