/**
 * @file 110A.CPP
 * @brief Solves the "Nearly Lucky Number" problem from Codeforces (Problem 110A).
 *        This program reads a large integer, counts the occurrences of "lucky digits"
 *        (4 and 7) within it, and then determines if this count itself is a lucky number.
 *        This version processes the number using arithmetic operations.
 * Algorithm: Digit Extraction and Counting
 * Time Complexity: O(log10 N), where N is the input number (proportional to the number of digits).
 * Space Complexity: O(1).
 */

#include <iostream> // Required for input/output operations (cin, cout).
using namespace std;

/**
 * @brief Main function to check if an input number is a "nearly lucky number".
 * @return 0 if the program executes successfully.
 */
int main() {
  long long int input, temp; // Declare long long integers for input and a temporary variable.
  // Read the input number from standard input.
  cin >> input;
  
  // 'flag' is declared but not used. 'count' is the actual counter for lucky digits.
  int flag = 0; 
  long long int count = 0;
  
  temp = input; // Initialize temp with the input number for digit extraction.
  
  /**
   * Block Logic: Extracts digits from 'temp' and counts occurrences of '4' or '7'.
   * Invariant: 'temp' is progressively reduced by removing its last digit, 'count' accumulates lucky digit count.
   */
  while (temp) { // Loop until all digits of 'temp' have been processed.
    int r = temp % 10; // Inline: Get the last digit of 'temp'.
    // Check if the extracted digit 'r' is a lucky digit (4 or 7).
    if (r == 7 || r == 4)
      count++; // If it is, increment the lucky digit counter.
    temp /= 10; // Inline: Remove the last digit from 'temp'.
  }

  /**
   * Block Logic: Determines if the count of lucky digits ('count') is itself a lucky number.
   * Pre-condition: 'count' holds the total count of lucky digits in the input number.
   * Post-condition: Prints "YES" if 'count' is 4 or 7, otherwise prints "NO".
   */
  if (count == 7 || count == 4) { // Check if the count 'count' is 4 or 7.
    cout << "YES";                // If so, the number is "nearly lucky".
    ; // Empty statement, likely a leftover from development.
  } else
    cout << "NO";                 // Otherwise, it is not.
    
  cout << endl; // Add newline for consistent output formatting.
  return 0; // Indicate successful program termination.
}
