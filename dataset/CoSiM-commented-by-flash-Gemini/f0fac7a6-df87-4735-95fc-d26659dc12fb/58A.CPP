/**
 * @file 58A.CPP
 * @brief This program determines if the word "hello" can be formed as a subsequence
 * from a given input string. A subsequence means the characters appear in the correct
 * order, but not necessarily consecutively.
 * Algorithm: Sequential character search. It uses multiple `for` loops, each responsible
 * for finding the next character of "hello" in the input string, starting the search
 * from where the previous character was found.
 * Time Complexity: O(N) where N is the length of the input string, as it performs
 * a single pass (though logically segmented) over the string.
 * Space Complexity: O(N) for storing the input string.
 */
#include <iostream> // Required for input/output operations (std::cin, std::cout).
#include <string>   // Required for std::string.

using namespace std;

int main() {
  string str; // Stores the input string.
  // Reads the input string from standard input.
  cin >> str;
  int i = 0;   // General loop index, re-used across sequential loops.
  int ctr = 0; // Counter to track how many characters of "hello" have been found in sequence.

  /**
   * Block Logic: Searches for the character 'h'.
   * The loop iterates from the current position of 'i'. If 'h' is found,
   * 'ctr' is incremented, 'i' is advanced, and the loop breaks to move to the next search stage.
   */
  for (; i < str.size(); i++)
    if (str[i] == 'h') {
      i++;   // Advance index to start next search from here.
      ctr++; // Increment found characters count.
      break; // Stop searching for 'h'.
    }
  /**
   * Block Logic: Searches for the character 'e'.
   * Resumes search from the position after 'h' was found.
   */
  for (; i < str.size(); i++)
    if (str[i] == 'e') {
      i++;
      ctr++;
      break;
    }
  /**
   * Block Logic: Searches for the first 'l'.
   * Resumes search from the position after 'e' was found.
   */
  for (; i < str.size(); i++)
    if (str[i] == 'l') {
      i++;
      ctr++;
      break;
    }
  /**
   * Block Logic: Searches for the second 'l'.
   * Resumes search from the position after the first 'l' was found.
   */
  for (; i < str.size(); i++)
    if (str[i] == 'l') {
      i++;
      ctr++;
      break;
    }
  /**
   * Block Logic: Searches for the character 'o'.
   * Resumes search from the position after the second 'l' was found.
   */
  for (; i < str.size(); i++)
    if (str[i] == 'o') {
      i++;
      ctr++;
      break;
    }
  // Prints "YES" if all 5 characters ('h', 'e', 'l', 'l', 'o') were found in sequence (i.e., 'ctr' is 5).
  // Otherwise, prints "NO".
  cout << (ctr >= 5 ? "YES" : "NO");
  // Standard program exit.
  return 0;
}