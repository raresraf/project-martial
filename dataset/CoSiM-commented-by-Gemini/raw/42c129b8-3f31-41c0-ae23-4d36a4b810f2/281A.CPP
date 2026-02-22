/**
 * @file 281A.CPP
 * @brief Solution to Codeforces problem 281A - Word Capitalization.
 *
 * @algorithm
 * The problem requires capitalizing the first letter of a given word.
 *
 * This solution processes the first character separately from the rest of the string.
 *
 * The algorithm is as follows:
 * 1. Read the input word into a C-style character array `ch`.
 * 2. Manually check if the first character `ch[0]` is a lowercase letter by
 *    comparing its ASCII value to the range [97, 122].
 * 3. If it is lowercase, convert it to uppercase by subtracting 32 from its
 *    ASCII value and print the resulting character.
 * 4. If it is not lowercase, print it as is.
 * 5. Iterate through the rest of the string from the second character to the
 *    end and print each character.
 *
 * @note The logic for handling the first character is unnecessarily complex,
 *       using a single-iteration `for` loop with `break` statements instead of a
 *       simple if-else block.
 *
 * @complexity
 * - Time Complexity: O(L), where L is the length of the word. `strlen` takes O(L),
 *   and the final print loop also takes O(L).
 * - Space Complexity: O(L) to store the word in the character array.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int d, c = 0, n;
  char ch[1000];
  cin.getline(ch, 1000);
  d = strlen(ch);
  /**
   * @brief This loop processes only the first character of the string to capitalize it.
   * Note: The loop structure `for(int i=0; i<1; i++)` is redundant; a simple
   * if-else block would be clearer.
   */
  for (int i = 0; i < 1; i++) {
    // Check if the first character is a lowercase letter (ASCII 'a' to 'z').
    if (ch[0] >= 97 && ch[0] <= 122) {
      // Manually convert to uppercase by adjusting ASCII value.
      n = ch[0] - 32;
      printf("%c", n);
      break; // Redundant break.
    } else {
      // If already capitalized or not a letter, print as is.
      cout << ch[0];
      break; // Redundant break.
    }
  }
  /**
   * @brief Print the rest of the string from the second character onwards.
   */
  for (int i = 1; i < d; i++) {
    cout << ch[i];
  }
  cout << endl;
  return 0;
}
