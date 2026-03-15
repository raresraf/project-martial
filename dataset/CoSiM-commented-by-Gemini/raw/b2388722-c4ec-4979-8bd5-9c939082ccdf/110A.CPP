/**
 * @file 110A.CPP
 * @brief Solution to the "Nearly Lucky Number" problem.
 * @details This program determines if a given number is "nearly lucky". A number is nearly lucky
 * if the count of its lucky digits ('4' and '7') is itself a lucky number (i.e., 4 or 7).
 * The program is written in a C-style, using C-style strings and I/O.
 *
 * @algorithm
 * 1. Reads the input number as a C-style string `s`. Reading as a string is necessary
 *    to handle numbers larger than standard integer types can hold.
 * 2. Initializes a counter `count` to zero.
 * 3. Iterates through the string character by character.
 * 4. For each character, it checks if it is a '4' or a '7'. If so, `count` is incremented.
 * 5. After the loop, a `switch` statement checks the final value of `count`.
 * 6. If `count` is 4 or 7, it prints "YES"; otherwise, it prints "NO".
 *
 * @complexity
 * - Time: O(L), where L is the number of digits in the input number, due to the single pass
 *   over the string.
 * - Space: O(L) to store the input number in a character array.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // A counter for the number of lucky digits ('4' or '7') found.
  int count = 0;
  // A C-style character array to store the input number as a string.
  // Size 21 is sufficient for a number up to 10^18 (which has at most 19 digits).
  char s[21];
  scanf("%s", s);

  /**
   * @brief This loop iterates through the digits of the number and counts the lucky ones.
   */
  for (int i = 0; i < strlen(s); i++)
    if (s[i] == '4' || s[i] == '7')
      count++;
  
  /**
   * @brief This switch statement checks if the count of lucky digits is itself a lucky number.
   * The problem defines lucky numbers as 4 and 7.
   */
  switch (count) {
  // The fall-through from case 4 to case 7 handles both lucky possibilities.
  case 4:
  case 7:
    printf("YES\n");
    break;
  default:
    printf("NO\n");
  }
  return 0;
}