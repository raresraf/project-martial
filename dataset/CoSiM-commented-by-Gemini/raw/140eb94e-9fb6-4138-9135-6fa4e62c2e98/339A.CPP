/**
 * @file 339A.CPP
 * @brief Sorts the numbers in a string representing a sum.
 * @details This program takes a string representing a sum of single-digit numbers (e.g., "3+2+1")
 * and reorders the numbers in non-decreasing order (e.g., "1+2+3"). The '+' signs remain in their positions.
 *
 * Algorithm:
 * 1. The input string is read into a character array.
 * 2. The numbers are located at even indices (0, 2, 4, ...) and the '+' signs at odd indices.
 * 3. An insertion sort algorithm is applied directly to the character array, but only considering
 *    the elements at even indices (the numbers).
 * 4. The inner loop compares the current number `a[i]` with the preceding numbers `a[j]` and swaps them
 *    to maintain sorted order.
 * 5. The final sorted string is printed.
 *
 * Time Complexity: O(L^2), where L is the length of the string, due to the nested loops for sorting.
 * Space Complexity: O(1), apart from the storage for the input string itself.
 */
#include <iostream>
using namespace std;

/**
 * @brief Calculates the length of a null-terminated C-style string.
 * @param a The character array.
 * @return The length of the string, excluding the null terminator.
 */
int strlength(char a[]) {
  int count = 0;
  while (a[count] != '\0')
    count++;
  return count;
}
int main() {
  char a[100];
  char temp;
  cin >> a;
  int i, j, len;
  len = strlength(a);
  //	cout << len <<endl;
  
  /**
   * @brief Sorts the numbers within the string using a variant of insertion sort.
   * The loops are designed to only operate on the characters at even indices, which
   * correspond to the positions of the numbers in the sum.
   */
  for (i = 2; i < len; i = i + 2) {
    // cout << a[i] << endl;
    for (j = 0; j < i; j = j + 2) {
      // cout << a[j] << endl;
      // If the number at the current position 'i' is less than a number
      // at a preceding position 'j', a swap is performed.
      if (a[i] < a[j]) {
        temp = a[i];
        a[i] = a[j];
        a[j] = temp;
        // cout << a[j] << a[i] << endl;
      }
    }
  }
  // Print the modified string, which now has its numbers sorted.
  cout << a << endl;
  return 0;
}
