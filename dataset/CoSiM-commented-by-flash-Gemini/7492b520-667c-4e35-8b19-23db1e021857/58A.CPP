/**
 * @file 58A.CPP
 * @brief This program solves the "Chat room" problem (Codeforces 58A).
 *   It determines if the word "hello" can be formed as a subsequence of a given input string.
 *   This means the characters 'h', 'e', 'l', 'l', 'o' must appear in the input string
 *   in that exact order, but not necessarily contiguously.
 *   Problem description: https://codeforces.com/problemset/problem/58/A
 * * Algorithm: Two-pointer linear scan.
 *   One pointer (`a`) tracks the current character needed from the target word "hello".
 *   Another pointer (`b`) scans the input string. When a match is found, the "hello"
 *   pointer advances. If the "hello" pointer successfully traverses all characters
 *   of "hello", then it is a subsequence.
 * Time Complexity: O(L) where L is the length of the input string (`line`),
 *   as both pointers iterate at most once through their respective strings.
 * Space Complexity: O(L) for storing the input `line` string.
 */
#include <iostream> // Required for standard input/output operations.
#include <string>   // Required for std::string and std::getline.
using namespace std;

/**
 * @brief Main function to check if "hello" is a subsequence of the input string.
 * Functional Utility: Reads an input string, performs a two-pointer scan to find "hello" as a subsequence,
 *                     and prints "YES" or "NO" based on the result.
 */
int main() {
  const string greeting = "hello"; // The target word to find as a subsequence.
  string line; // Stores the input string from the user.
  // Block Logic: Reads an entire line of input into the `line` string.
  getline(cin, line);
  int a = 0; // Pointer/index for the `greeting` string (tracks which character of "hello" is currently being searched for).
  int b = 0; // Pointer/index for the `line` string (scans through the input).
  string output = "NO"; // Stores the result, initialized to "NO".
  /**
   * Block Logic: Implements a two-pointer linear scan to find `greeting` as a subsequence in `line`.
   * Pre-condition: `a` points to the start of `greeting`, `b` points to the start of `line`.
   * Invariant: `b` always advances through `line`. `a` advances only when a match is found.
   */
  while (b < line.size()) {
    // Block Logic: Checks if the character at the current position in `line` (`line[b]`) matches
    //              the character currently being sought from `greeting` (`greeting[a]`).
    // Inline: `b++` increments the `line` pointer after checking the character.
    if (line[b++] == greeting[a]) {
      a++; // If a match is found, advance the `greeting` pointer.
    }
    // Block Logic: Checks if all characters of `greeting` have been found in order.
    // If `a` has advanced beyond the last index of `greeting`, it means "hello" has been found.
    if (a >= greeting.size()) {
      output = "YES"; // Sets the output to "YES".
      break;          // Exits the loop early as the subsequence has been found.
    }
  }
  // Block Logic: Prints the final determined output ("YES" or "NO").
  cout << output << endl;
  return 0;
}
