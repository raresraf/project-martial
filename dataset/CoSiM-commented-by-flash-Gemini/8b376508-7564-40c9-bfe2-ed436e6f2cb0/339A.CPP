/**
 * @8b376508-7564-40c9-bfe2-ed436e6f2cb0/339A.CPP
 * @brief This program reorders a given sum of 1s, 2s, and 3s into a non-decreasing order.
 * For example, "3+2+1" becomes "1+2+3". It parses the input string, counts the occurrences
 * of each number, and then prints them in ascending order separated by '+' signs.
 * Problem Link: https://codeforces.com/problemset/problem/339/A
 * Algorithm: Counting sort (specifically, counting occurrences of small, fixed-range integers).
 * Time Complexity: O(L), where L is the length of the input sum string, due to two linear passes.
 * Space Complexity: O(1) as the number of distinct integers (1, 2, 3) is constant.
 */

#include <iostream> // For input/output operations
#include <string>   // For string manipulation

using namespace std;

int main() {
  // @var sum: Stores the input string representing the sum (e.g., "1+2+3").
  string sum;
  cin >> sum; // Read the input sum string.

  // @var nums: An array to store the counts of numbers 1, 2, and 3.
  // nums[0] for count of 1s, nums[1] for 2s, nums[2] for 3s.
  int nums[3];
  nums[0] = nums[1] = nums[2] = 0; // Initialize counts to zero.
  
  // @var num: Temporary variable to store the integer value of a character.
  int num;
  
  // Block Logic: Parse the input string and count occurrences of each number.
  // Invariant: 'sum[i]' is always a digit, 'sum[i+1]' is always a '+'.
  // Pre-condition: Input string alternates between digits and '+' characters.
  for (int i = 0; i < sum.length(); i += 2) {
    // Inline: Convert character digit to its integer value. '0' is ASCII 48.
    num = sum[i] - '0';
    // Inline: Increment the count for the corresponding number (1->nums[0], 2->nums[1], 3->nums[2]).
    nums[num - 1]++;
  }

  // @var first: A boolean flag to handle printing '+' only between numbers.
  bool first = true;
  
  // Block Logic: Iterate through the counts and print the numbers in non-decreasing order.
  // Invariant: Numbers are printed from smallest to largest (1s, then 2s, then 3s).
  // Pre-condition: 'nums' array contains the correct counts of each number.
  for (int i = 0; i < 3; i++) { // Iterate for numbers 1, 2, 3 (represented by indices 0, 1, 2).
    // Block Logic: Print each number according to its count.
    for (int j = 0; j < nums[i]; j++) {
      // Block Logic: Print '+' separator before each subsequent number.
      if (!first) {
        cout << "+";
      }
      // Inline: Print the actual number (index + 1).
      cout << i + 1;
      // Inline: After printing the first number, set 'first' to false.
      first = false;
    }
  }
  // Post-condition: Print a newline character at the end of the output.
  cout << "
";
  return 0; // Explicitly return 0 for successful execution.
}
