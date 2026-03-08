/**
 * @file 339A.CPP
 * @brief Sorts a sum of numbers (1, 2, 3) given as a string.
 * @note The original code includes a link to the problem statement:
 * https://codeforces.com/problemset/problem/339/A
 *
 * @details
 * This program reads a string representing a sum, like "3+2+1", and rearranges
 * it into non-decreasing order, like "1+2+3". It uses a counting sort algorithm,
 * which is highly efficient for a small, fixed range of numbers. First, it counts
 * the occurrences of '1', '2', and '3'. Then, it reconstructs the string by
 * printing the numbers in sorted order, separated by plus signs.
 *
 * @section
 * Algorithm: Counting Sort.
 * Time Complexity: O(N), where N is the length of the input string. The solution involves one pass to count and one pass to print.
 * Space Complexity: O(1), as the frequency map `nums` has a fixed size.
 */
#include <iostream>

using namespace std;

int main() {
  // 'sum' stores the input string, e.g., "1+3+2".
  string sum;
  cin >> sum;

  // 'nums' is a frequency map. nums[0] counts 1s, nums[1] counts 2s, nums[2] counts 3s.
  int nums[3];
  nums[0] = nums[1] = nums[2] = 0;
  int num;
  /**
   * @brief First pass: Count the occurrences of each number.
   *
   * Logic: This loop iterates over the string, stepping by 2 to skip the '+'
   * signs. It converts each number character to an integer and increments the
   * corresponding counter in the 'nums' array.
   */
  for (int i = 0; i < sum.length(); i += 2) {
    num = sum[i] - '0'; // Convert char '1','2','3' to int 1,2,3.
    nums[num - 1]++;    // Increment the frequency count for that number.
  }

  // 'first' is a flag to handle the placement of '+' separators correctly.
  bool first = true;
  /**
   * @brief Second pass: Reconstruct and print the sorted sum.
   *
   * Logic: The outer loop iterates through the numbers 1, 2, and 3 (as i=0 to 2).
   * The inner loop prints the current number as many times as it was counted.
   * A '+' is printed before each number except the very first one.
   */
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < nums[i]; j++) {
      // Pre-condition: If this isn't the first number printed, add a separator.
      if (!first) {
        cout << "+";
      }
      // Print the number itself (i=0 -> 1, i=1 -> 2, i=2 -> 3).
      cout << i + 1;
      // After the first number is printed, all subsequent numbers need a '+'.
      first = false;
    }
  }
  cout << "
";
}
