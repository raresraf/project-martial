/**
 * @file 58A.CPP
 * @brief Subsequence search problem ("Chat room").
 *
 * @details
 * This program determines if the word "hello" can be formed as a subsequence
 * from a given input string. It reads a string and iterates through it, checking
 * if the characters 'h', 'e', 'l', 'l', 'o' appear in that specific order,
 * though not necessarily contiguously.
 *
 * @section
 * Algorithm: Two-pointer subsequence search.
 * One pointer traverses the input string, and another pointer traverses the
 * target subsequence ("hello"). The subsequence pointer only advances when a
 * matching character is found.
 *
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(L) for storing the input string.
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Attempts to initialize a "hello" string. This function is unused.
 * @warning This function is critically flawed. It returns a pointer to a local
 * variable (`tmp`), which is destroyed upon function exit. Using the returned
 * pointer would result in undefined behavior.
 * @return A dangling pointer to what was the local "hello" string.
 */
char *init_hello() {
  char tmp[6] = "hello";
  return tmp;
}

/**
 * @brief Allocates memory and reads a string from standard input. This function is unused.
 * @return A pointer to a newly allocated char array containing the input string.
 */
char *init_chatroot_io() {
  char *s = (char *)malloc(sizeof(char) * 100);
  cin >> s;
  return s;
}

int main() {
  // The target subsequence to be found.
  char t[] = "hello";
  // 'hello' is a pointer that tracks the current character we are looking for in the target.
  char *hello = t;
  // 's' is a pointer to the input string.
  char *s = (char *)malloc(sizeof(char) * 100);
  cin >> s;
  // 'count' tracks how many characters of the "hello" subsequence have been found.
  int count = 0;

  /**
   * @brief Main loop to find the subsequence.
   *
   * Logic: This loop implements a two-pointer algorithm. The pointer 's' always
   * moves forward through the input string. The pointer 'hello' only moves
   * forward when the character it's pointing to is found at the current
   * position of 's'.
   */
  while (*s) {
    // Pre-condition: Check if the current character in the input string matches
    // the current character we're searching for in "hello".
    if (*s == *hello) {
      // Invariant: A character of the subsequence has been found in order.
      // Advance the 'hello' pointer to look for the next character.
      hello++;
      count++;
    }
    // Always advance the input string pointer.
    s++;
  }

  // Post-condition: If 'count' is 5, all characters of "hello" were found in order.
  if (count == 5)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
