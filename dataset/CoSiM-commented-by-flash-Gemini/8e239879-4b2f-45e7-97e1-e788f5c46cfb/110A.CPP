/**
 * @8e239879-4b2f-45e7-97e1-e788f5c46cfb/110A.CPP
 * @brief This program determines if a given number is a "Nearly Lucky Number"
 * according to specific rules. A number is nearly lucky if the count of its
 * lucky digits (4s and 7s) is itself a lucky number (4 or 7).
 * Problem Link: https://codeforces.com/problemset/problem/110/A
 * Algorithm: Digit extraction and counting, followed by conditional check.
 * Time Complexity: O(log10 N), where N is the input number, as it iterates through its digits.
 * Space Complexity: O(1) for storing variables.
 */

#include <iostream> // Required for standard input/output operations (cin, cout).

using namespace std;

int main() {
  // @var n: The input number to be checked for "nearly lucky" property.
  long long n;
  cin >> n; // Read the input number.
  
  // @var count: Counter for the number of "lucky digits" (4s and 7s) in 'n'.
  int count = 0;
  
  // Block Logic: Extract digits from 'n' and count occurrences of 4s and 7s.
  // Invariant: 'n' is progressively reduced by removing its last digit.
  // Pre-condition: 'n' is a non-negative integer.
  while (n != 0) {
    // Block Logic: Check if the last digit of 'n' is 4 or 7.
    if (n % 10 == 4 || n % 10 == 7) {
      count += 1; // Increment the lucky digit counter.
    }
    n /= 10; // Remove the last digit from 'n'.
  }
  
  // Block Logic: Determine if the count of lucky digits is itself a lucky number.
  // If 'count' is 4 or 7, then the original number is "nearly lucky".
  // Post-condition: Print "YES" or "NO" based on the nearly lucky condition.
  if (count == 4 || count == 7) {
    cout << "YES" << endl; // Output "YES".
  } else {
    cout << "NO" << endl;  // Output "NO".
  }
  return 0; // Explicitly return 0 for successful execution.
}
