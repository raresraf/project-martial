/**
 * @file 266A.CPP
 * @brief Counts the number of adjacent identical items in a sequence.
 *        This solves the "Stones on the Table" problem, determining how many items need to be removed
 *        so that no two adjacent items are the same.
 *
 * Algorithm: Iterative comparison of consecutive elements. Each item `a[i]` is compared with its immediate predecessor `a[i-1]`.
 *            The count increments for every pair of identical adjacent elements.
 * Time Complexity: O(N) where N is the number of items in the sequence.
 * Space Complexity: O(N) for storing the sequence of items.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.

using namespace std;

/**
 * @brief Main function to read a sequence of items and count adjacent identical ones.
 * @pre Input consists of an integer 'n' (number of items), followed by 'n' characters (or single-character strings).
 * @post Prints the total count of adjacent identical items.
 */
int main() {
  char a[101]; // Array to store up to 100 characters (+1 for potential 0-indexing usage or null terminator if %s reads string).
               // The loops use 1-based indexing for 'i' up to 'n', effectively using a[1] to a[n].
  int i, n, j = 0, count = 0; // 'n': number of items, 'j': index for the previous item to compare against, 'count': adjacent identical items.

  // Functional Utility: Reads the total number of items 'n'.
  scanf("%d", &n);

  /**
   * Block Logic: Reads 'n' items (characters) into the array 'a', starting from index 1.
   * Note: Using `scanf("%s", &a[i]);` to read single characters might be a simplification
   *       for single-character inputs. It implicitly reads one character into `a[i]` and overwrites the null terminator if it exists.
   *       For single characters, `scanf(" %c", &a[i]);` or `cin >> a[i];` would be more explicit.
   * Invariant: After this loop, 'a' contains the input sequence from `a[1]` to `a[n]`.
   */
  for (i = 1; i <= n; i++) {
    scanf("%s", &a[i]); // Reads a string. If the input is a single character, it effectively reads that character.
  }

  // Functional Utility: Initializes 'a[0]' to a character that won't match common input ('a', 'b', 'c', 'R', 'G', 'B', etc.).
  //                     This acts as a sentinel for the first comparison with `a[1]`, ensuring it's not counted as a duplicate.
  //                     The chosen value 'a' implicitly means it's unlikely to match the actual game characters.
  a[j] = 'a'; // j is initialized to 0. So, a[0] is set to 'a'.

  /**
   * Block Logic: Iterates through the input sequence (from `a[1]` to `a[n]`), comparing each item with its direct predecessor.
   * Invariant: `j` always holds the index of the character immediately preceding `a[i]` in the original sequence
   *            that is considered for comparison (i.e., `j = i-1`).
   *            'count' accumulates the total number of times `a[i]` is identical to `a[i-1]`.
   */
  for (i = 1; i <= n; i++) {
    /**
     * Block Logic: Performs a direct comparison between the current item and its effective predecessor.
     * Pre-condition: `a[i]` is the current item, and `a[j]` (which is `a[i-1]`) is its predecessor.
     * Invariant: If `a[i]` and `a[j]` are identical, it signifies an adjacent duplicate, and `count` is incremented.
     *            `j` is incremented in both branches to always point to the index of the character that was just processed (which becomes the predecessor for the next iteration).
     */
    if (a[i] == a[j]) { // Checks if the current item is identical to the preceding item.
      count++;          // Increments count if an adjacent duplicate is found.
      j++;              // Moves the predecessor index forward to match the current 'i'.
    } else {
      j++; // Moves the predecessor index forward to match the current 'i'.
    }
  }

  // Functional Utility: Prints the final total count of adjacent identical items.
  cout << count << endl;
  // Indicates successful program execution.
  return 0;
}
