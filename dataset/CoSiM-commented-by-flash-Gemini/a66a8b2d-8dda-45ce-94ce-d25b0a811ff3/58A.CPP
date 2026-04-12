/**
 * @file 58A.CPP
 * @brief Subsequence detection algorithm for identifying target patterns in noisy streams.
 * 
 * Logic: Determines if the target sequence "hello" is present as a subsequence 
 * within the input string 'a'. Characters must appear in the correct order, 
 * though they may be non-contiguous.
 * 
 * Algorithm: Greedy linear scan with single-pointer progression on the target pattern.
 * Time Complexity: $O(N)$ where N is the input string length.
 * Space Complexity: $O(N)$ for string storage.
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  // a: Input sequence, b: Target pattern ("hello").
  string a, b = "hello";
  cin >> a;
  
  // j: Pointer into the target sequence, cont: Counter for matched characters.
  int i, j = 0, cont = 0;

  /**
   * Block Logic: Main scanning loop.
   * Invariant: 'cont' represents the length of the prefix of "hello" found so far.
   */
  for (i = 0; i < a.length(); i++) {
    /**
     * Logic: If current character matches the next required character in "hello".
     */
    if (a[i] == b[j]) {
      j++;    // Advance target pointer.
      cont++; // Increment match count.
    }
    
    // Optimization: Early termination once the entire target sequence is identified.
    if (cont == 5)
      break;
  }

  /**
   * Functional Utility: Final state evaluation.
   */
  if (cont == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
    
  return 0;
}
