/**
 * @file 58A.CPP
 * @brief Solves the "Chat room" problem from Codeforces (Problem 58A).
 *        This program determines if the word "hello" can be formed as a subsequence
 *        from a given input string.
 * Algorithm: Subsequence Matching (Greedy Approach)
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(1) (excluding input/output storage for the string itself).
 */

#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Initializes and returns a pointer to a local character array "hello".
 * @warning This function returns a pointer to a local stack variable,
 *          which will lead to undefined behavior when accessed after the function returns.
 *          This is a critical bug in the original code.
 * @return A pointer to a character array containing "hello".
 */
char *init_hello() {
  char tmp[6] = "hello";
  return tmp;
}

/**
 * @brief Reads an input string from stdin and allocates memory for it.
 * @warning The allocated memory is not freed within this function, leading to a memory leak.
 *          Additionally, using `cin >> s` for a char array is prone to buffer overflows.
 * @return A pointer to the dynamically allocated character array containing the input string.
 */
char *init_chatroot_io() {

  char *s = (char *)malloc(sizeof(char) * 100);
  cin >> s;
  return s;
}

/**
 * @brief Main function to read an input string and check if "hello" is its subsequence.
 * @return 0 if the program executes successfully.
 */
int main() {
  // Target subsequence: "hello". 't' is a stack-allocated array.
  char t[] = "hello";
  // 'hello' pointer points to the beginning of the target subsequence.
  char *hello = t;
  // Dynamically allocate memory for the input string.
  // Using a fixed-size buffer can lead to buffer overflows if input is too long.
  char *s = (char *)malloc(sizeof(char) * 100);
  // Read the input string from standard input.
  cin >> s;
  // Counter to track how many characters of "hello" have been matched.
  int count = 0;
  
  /**
   * Block Logic: Iterates through the input string to find "hello" as a subsequence.
   * Invariant: '*s' points to the current character being examined in the input string.
   *            '*hello' points to the next character of the target "hello" subsequence to find.
   */
  while (*s) { // Loop until the end of the input string is reached.
    // Check if the current character in the input string matches the next character needed for "hello".
    if (*s == *hello) {
      // Inline: Move to the next character in the target "hello" subsequence.
      *hello++;
      // Increment the match count.
      count++;
      // Inline: Move to the next character in the input string.
      *s++;
    } else
      // Inline: If no match, only move to the next character in the input string.
      *s++;
  }
  
  /**
   * Block Logic: Determines and prints whether "hello" was found as a subsequence.
   * Pre-condition: 'count' holds the number of matched characters from "hello".
   * Post-condition: Prints "YES" if all 5 characters of "hello" were found, "NO" otherwise.
   */
  if (count == 5) // If all 5 characters of "hello" were found in order.
    cout << "YES"; // Output "YES".
  else
    cout << "NO"; // Output "NO".
  
  // Free the dynamically allocated memory for the input string.
  free(s); // This was missing in the original code, causing a memory leak.

  return 0; // Indicate successful execution.
}