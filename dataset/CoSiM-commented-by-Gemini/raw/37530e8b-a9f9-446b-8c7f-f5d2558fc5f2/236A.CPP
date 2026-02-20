/**
 * @file 236A.CPP
 * @brief Solves the "Boy or Girl" problem by sorting and counting distinct characters.
 *
 * This program determines if a user is male or female based on the number of
 * distinct characters in their username. An even number of distinct characters
 * indicates a female user ("CHAT WITH HER!"), while an odd number indicates a male
 * user ("IGNORE HIM!").
 *
 * This implementation first sorts the username string, which groups all identical
 * characters together. It then iterates through the sorted string to count the
 * number of unique characters.
 *
 * Algorithm:
 * 1. Read the username into a C-style character array.
 * 2. Sort the character array alphabetically.
 * 3. Iterate through the sorted array, counting the number of times a character
 *    is different from the one immediately following it. This implicitly counts
 *    the number of unique characters by counting the "boundaries" between
 *    different character groups, including the boundary with the final null terminator.
 * 4. Check if the resulting count is even or odd and print the appropriate message.
 *
 * Time Complexity: O(L log L), dominated by the sorting operation on the string of length L.
 * Space Complexity: O(L) for the character array buffer.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  char s[100];
  cin >> s;
  int i, j, c = 0; // c: counter for distinct characters.
  int l = strlen(s);

  // Sort the character array to group identical characters together.
  sort(s, s + l);

  /**
   * Block Logic: Count the number of distinct characters in the sorted string.
   * This loop increments the counter 'c' each time it finds a character that
   * is different from the next one. This works because after sorting, all
   * identical characters are contiguous. The final character of the string
   * will be compared against the null terminator (`\0`), which correctly
   * increments the count for the last group of characters.
   */
  for (i = 0; i < l; i++) {
    if (s[i] != s[i + 1]) {
      c++;
    }
  }

  // Check if the number of distinct characters is even or odd.
  if (c % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else
    cout << "IGNORE HIM!";
}
