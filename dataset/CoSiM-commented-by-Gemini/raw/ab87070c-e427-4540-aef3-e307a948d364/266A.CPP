/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem from Codeforces.
 * @details The program calculates the minimum number of stones to remove from a row
 * such that no two adjacent stones have the same color. This is equivalent to
 * counting the number of adjacent pairs of stones with the same color.
 *
 * Algorithm:
 * 1. Read the number of stones, 'n' (which is not directly used in the logic).
 * 2. Read the sequence of stone colors as a string into a character array.
 * 3. Initialize a counter 'co' to 0.
 * 4. Iterate through the array from the first stone to the second-to-last stone.
 * 5. In each iteration, compare the current stone `arr[i]` with the next one `arr[i+1]`.
 * 6. If they are the same, increment the counter 'co'.
 * 7. After the loop, print the final count 'co'.
 *
 * Time Complexity: O(L), where L is the number of stones (the length of the string),
 * due to the single pass for comparison.
 * Space Complexity: O(L) to store the sequence of stone colors.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, i, co = 0;
  scanf("%d", &n);
  getchar();
  char arr[51];
  cin.getline(arr, 51);
  int len = strlen(arr);
  /**
   * Block Logic: Iterate through the stones and count adjacent duplicates.
   * The loop runs up to the second-to-last element to prevent out-of-bounds access.
   */
  for (i = 0; i < len - 1; i++) {
    // If a stone has the same color as the one next to it...
    if (arr[i] == arr[i + 1]) {
      // ...it needs to be removed, so increment the counter.
      co++;
    }
  }
  printf("%d\n", co);
}
