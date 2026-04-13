/**
 * @file 236A.CPP
 * @brief This program determines a message ("CHAT WITH HER!" or "IGNORE HIM!")
 * based on the parity (odd or even) of the number of unique characters in an input username string.
 * Algorithm: First, the input string is sorted using a bubble sort-like approach.
 * Then, a linear scan is performed on the sorted string to count the number of unique characters.
 * Time Complexity: O(N^2) for sorting (where N is the string length) + O(N) for counting unique characters.
 * Space Complexity: O(N) for storing the input string.
 */
#include <stdio.h>  // Required for standard input/output functions like scanf and printf.
#include <string.h> // Required for string manipulation functions like strlen.

int main() {
  int i, j, count = 0;             // Loop counters and 'count' for unique characters.
  char temp, ch[200];              // 'temp' for swapping characters during sort, 'ch' for the input string.
  // char temp1, ch1[200];         // 'temp1' and 'ch1' are declared but not used in this implementation.
  scanf("%s", ch); // Reads the input username string.

  /**
   * Block Logic: Sorts the characters of the input string 'ch' in descending order.
   * This is a basic bubble sort implementation. Sorting helps in easily counting unique
   * characters in the subsequent step by grouping identical characters together.
   * Invariant: After each pass of the outer loop, the largest unsorted character is at its correct position.
   */
  for (i = 0; i < strlen(ch); i++) {
    for (j = i + 1; j < strlen(ch); j++) {
      if (ch[j] > ch[i]) { // Compares characters for sorting.
        temp = ch[i];     // Swaps characters to achieve descending order.
        ch[i] = ch[j];
        ch[j] = temp;
      }
    }
  }

  // If the string is not empty, the first character is unique, so count starts at 1.
  if (strlen(ch) > 0) {
    count = 1;
    temp = ch[0]; // Stores the first character to compare against.
  }
  
  /**
   * Block Logic: Counts the number of unique characters in the sorted string.
   * It iterates through the sorted string and increments 'count' only when the current
   * character is different from the previously encountered unique character.
   * Invariant: 'count' accurately reflects the number of unique characters encountered so far.
   */
  for (i = 0; i < strlen(ch); i++) {
    if (ch[i] != temp) { // Checks if the current character is different from the previous unique one.
      count++;           // Increments unique character count.
      temp = ch[i];      // Updates 'temp' to the new unique character.
    }
  }

  /**
   * Conditional Logic: Determines the output message based on the parity of the unique character count.
   * If 'count' is even, it prints "CHAT WITH HER!".
   * If 'count' is odd, it prints "IGNORE HIM!".
   */
  if (count % 2 == 0)
    printf("CHAT WITH HER!");
  else
    printf("IGNORE HIM!");
  // Standard program exit.
  return 0;
}