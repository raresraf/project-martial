/**
 * @file 58A.CPP
 * @brief This program solves the "Chat room" problem (Codeforces 58A).
 *   It determines if the word "hello" can be formed as a subsequence of a given input string.
 *   This means the characters 'h', 'e', 'l', 'l', 'o' must appear in the input string
 *   in that exact order, but not necessarily contiguously.
 *   Problem description: https://codeforces.com/problemset/problem/58/A
 * * Algorithm: Two-pointer linear scan.
 *   One pointer (`j`) tracks the current character needed from the target word "hello".
 *   Another pointer (`i`) scans the input string. When a match is found, the "hello"
 *   pointer advances. If the "hello" pointer successfully traverses all characters
 *   of "hello", then it is a subsequence.
 * Time Complexity: O(L) where L is the length of the input string (`str`).
 *   Although `strlen(str)` is in the loop condition, typically C++ compilers or
 *   competitive programming setups optimize this to be O(L) overall.
 * Space Complexity: O(L) for storing the input `str`.
 */
#include <cstring>  // Required for `strlen`.
#include <iostream> // Required for `std::cin`, `std::cout`.

/**
 * @brief Main function to check if "hello" is a subsequence of the input string.
 * Functional Utility: Reads an input string, performs a two-pointer scan to find "hello" as a subsequence,
 *                     and prints "YES" or "NO" based on the result.
 */
int main() {
  int i, j, flag = 0; // `i`: scans input string, `j`: tracks "hello" characters, `flag`: indicates if "hello" is found.
  char str[103];      // C-style string to store the input string. Max length 102 characters + null terminator.
  char str2[5] = {'h', 'e', 'l', 'l', 'o'}; // C-style string for the target word "hello".
  // Block Logic: Reads the input string into `str`. `std::cin >> str` reads a single word (whitespace-separated).
  std::cin >> str;
  /**
   * Block Logic: Implements a two-pointer linear scan. `i` iterates through `str`, `j` advances through `str2`.
   * Pre-condition: `str` is populated with the input string.
   * Invariant: `j` increments only when a character from `str2` is matched sequentially in `str`.
   * Inline: `strlen(str)` computes the length of `str`. For efficiency in competitive programming,
   *         it's often better to call `strlen` once before the loop or use `std::string`.
   */
  for (i = 0, j = 0; i < strlen(str); i++) {
    // Conditional Logic: Checks if the current character in `str` matches the `j`-th character of `str2`.
    if (str[i] == str2[j])
      j++; // If a match is found, advance the "hello" pointer.
    // Conditional Logic: If all characters of "hello" have been found (`j` reaches 5).
    if (j == 5) {
      flag = 1;  // Set flag to indicate "hello" is a subsequence.
      break;     // Exit the loop early.
    }
  }
  // Conditional Logic: Prints the result based on the `flag`.
  if (flag == 1)
    std::cout << "YES" << '\n'; // Prints "YES" if "hello" is a subsequence.
  else if (flag == 0)
    std::cout << "NO" << '\n';  // Prints "NO" otherwise.
  return 0;
}
