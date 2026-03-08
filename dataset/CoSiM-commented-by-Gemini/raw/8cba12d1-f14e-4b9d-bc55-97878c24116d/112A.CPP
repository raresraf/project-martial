/**
 * @file 112A.CPP
 * @brief Solution for the "Petya and Strings" problem (case-insensitive string comparison).
 *
 * @details
 * This program compares two input strings lexicographically, ignoring the case of
 * the letters. It first converts both strings to lowercase. Then, it iterates
 * through the strings to find the first differing character and determines which
 * string is lexicographically smaller or if they are equal.
 *
 * @section
 * Algorithm:
 * 1. Read two strings.
 * 2. Convert both strings to lowercase in-place.
 * 3. Perform a character-by-character comparison, breaking at the first difference.
 * 4. Use flag variables to determine the result and print -1, 0, or 1.
 *
 * Time Complexity: O(L), where L is the length of the strings.
 * Space Complexity: O(L) for storing the two strings.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  string p;
  getline(cin, s);
  getline(cin, p);
  int len = s.size();
  /**
   * @brief This loop normalizes both strings to lowercase.
   *
   * Logic: It iterates through each character and, if it's an uppercase letter,
   * converts it to its lowercase equivalent using ASCII arithmetic.
   */
  for (int i = 0; i < len; i++) {
    if (s[i] >= 'A' && s[i] <= 'Z')
      s[i] = (s[i] - 'A') + 'a';
    if (p[i] >= 'A' && p[i] <= 'Z')
      p[i] = (p[i] - 'A') + 'a';
  }

  /* Note: The following large block of code was commented out in the original
   * source. It appears to be an implementation of a bubble sort for each string,
   * which is not relevant to solving the lexicographical comparison problem.
   *
   * for(int i = 0; i < len - 1; i++){
   *     bool sw = false;
   *     for(int j = 0; j < len - 1 - i; j++){
   *         if(s[j] > s[j + 1]){
   *             int temp = s[j];
   *             s[j] = s[j + 1];
   *             s[j + 1] = temp;
   *             sw = true;
   *         }
   *     }
   *     if(sw == false) break;
   * }
   *
   * for(int i = 0; i < len - 1; i++){
   *     bool sw = false;
   *     for(int j = 0; j < len - 1 - i; j++){
   *         if(p[j] > p[j + 1]){
   *             int temp = p[j];
   *             p[j] = p[j + 1];
   *             p[j + 1] = temp;
   *             sw = true;
   *         }
   *     }
   *     if(sw == false) break;
   * }
   */
  // c1 and c2 are used as flags to indicate the comparison result.
  // c1 > 0 means s > p. c2 > 0 means s < p.
  int c1 = 0, c2 = 0;

  /**
   * @brief This loop performs the lexicographical comparison.
   *
   * Logic: It iterates through the now-lowercase strings and finds the first
   * position where the characters are not equal. It sets a flag and breaks,
   * as the lexicographical order is determined by this first difference.
   */
  for (int i = 0; i < len; i++) {
    if (s[i] > p[i]) {
      c1++;
      break;
    } else if (s[i] < p[i]) {
      c2++;
      break;
    }
  }
  /**
   * @brief Prints the result based on the flag values.
   *
   * Logic: Due to the 'break' statements, at most one of c1 or c2 can be non-zero.
   * - If c1 is 1, s was greater than p.
   * - If c2 is 1, s was less than p.
   * - If both are 0, the loop finished without finding a difference, so they are equal.
   */
  if (c1 > c2)
    printf("1
");
  else if (c1 < c2)
    printf("-1
");
  else if (c1 == c2)
    printf("0
");

  return 0;
}
