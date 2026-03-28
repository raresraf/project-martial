/**
 * @file 58A.CPP
 * @brief Checks if "hello" is a subsequence of a given string.
 *
 * This program reads a line of text and determines if the word "hello" can be
 * formed by deleting zero or more characters from the input string.
 *
 * @b Algorithm: The program uses a greedy two-pointer approach. One pointer (`a`)
 *    tracks the current character in the target word "hello", and the other (`b`)
 *    iterates through the input string. The pointer `a` only advances when a
 *    matching character is found, ensuring the subsequence order is maintained.
 * @b Time_Complexity: O(L), where L is the length of the input string, as it
 *    requires only a single pass.
 * @b Space_Complexity: O(L) to store the input string.
 */
#include <iostream>
#include <string> // Included for clarity

using namespace std;

int main() {
  const string greeting = "hello";
  string line;
  getline(cin, line);
  
  int a = 0; // Pointer for the target `greeting` string.
  int b = 0; // Pointer for the input `line`.
  string output = "NO";
  
  /**
   * @brief Main loop to find the subsequence.
   * It scans the input `line` from left to right.
   */
  while (b < line.size()) {
    // Invariant: `a` points to the next character we need to find in "hello".
    if (line[b] == greeting[a]) {
      // We found the next character of the subsequence, so we advance `a`.
      a++;
    }
    // Always advance the pointer for the input string.
    b++;
    
    // If `a` has reached the length of "hello", all characters have been found.
    if (a >= greeting.size()) {
      output = "YES";
      break;
    }
  }
  
  cout << output << endl;
  return 0;
}