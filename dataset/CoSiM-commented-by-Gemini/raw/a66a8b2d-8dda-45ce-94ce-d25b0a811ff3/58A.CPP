/**
 * @file 58A.CPP
 * @brief Solution for CodeForces Problem 58A - "Chat room".
 * 
 * @details
 * This program determines if it's possible to spell the word "hello" by
 * deleting zero or more characters from a given input string. This is a
 * classic subsequence problem.
 *
 * @algorithm Two-Pointer Subsequence Check
 * 1. Define the target string "hello".
 * 2. Use one pointer (`j`) for the target string "hello" and another (`i`) for the input string `a`.
 * 3. Iterate through the input string `a` with pointer `i`.
 * 4. If the character `a[i]` matches the character `b[j]`, it means we have found the
 *    next character of "hello" in sequence. We then advance the pointer `j` for the target string.
 * 5. Keep a count (`cont`) of matched characters.
 * 6. If the count reaches 5, it means all characters of "hello" have been found in the correct
 *    order, so we can stop and declare success ("YES").
 * 7. If the end of the input string is reached and the count is less than 5, it's a failure ("NO").
 *
 * @time_complexity O(L), where L is the length of the input string `a`.
 * @space_complexity O(1), as the target string "hello" has a fixed size.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  string a;      // Input string.
  string b = "hello"; // The target subsequence.
  cin >> a;
  
  int j = 0;    // Pointer for the target string `b`.
  int cont = 0; // Counter for matched characters.

  /**
   * @brief Main loop to find the "hello" subsequence.
   * This loop iterates through the input string `a` to find characters
   * that match the target string `b` in the correct order.
   */
  for (int i = 0; i < a.length(); i++) {
    // Pre-condition: Check if the current character in `a` matches the expected character in `b`.
    if (a[i] == b[j]) {
      j++;    // Move to the next character in the target string "hello".
      cont++; // Increment the count of matched characters.
    }
    
    // Optimization: if all 5 characters are found, no need to scan the rest of the string.
    if (cont == 5) {
      break;
    }
  }

  // If we found all 5 characters, it's a "YES".
  if (cont == 5) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  
  return 0;
}
