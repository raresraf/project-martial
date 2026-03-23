/**
 * @file 110A.CPP
 * @brief This competitive programming solution determines if a given number
 *        is considered "nearly lucky". A number is defined as "nearly lucky"
 *        if the count of its lucky digits (4s and 7s) is itself a lucky number
 *        (i.e., its digits are only '4's and '7's).
 * Algorithm: Digit Extraction and Counting using Integer Arithmetic.
 *            1. Count the occurrences of '4's and '7's in the input number `n`
 *               by repeatedly taking modulo 10 and dividing by 10. Store this count in `cnt`.
 *            2. Then, check the digits of `cnt` itself. If `cnt` contains any digit
 *               other than '4' or '7', then the original number is not "nearly lucky".
 * Time Complexity: O(log10(N)), where N is the input number. This is because
 *                  the number of operations is proportional to the number of digits
 *                  in N and then the number of digits in the count of lucky digits.
 * Space Complexity: O(1), as only a few fixed-size integer variables are used.
 */
#include <bits/stdc++.h> // Includes common standard library headers, including iostream.
using namespace std;

/**
 * @brief Main function of the program. Determines if an input number is "nearly lucky".
 * Functional Utility: Serves as the entry point for execution, implementing the core logic
 *                     to classify numbers based on the count and composition of their lucky digits.
 */
int main() {
  long long int n;      // Declares a long long integer to hold the input number.
  long long int cnt = 0; // Counter for lucky digits (4s and 7s) in the input number `n`.
  bool lucky = true;    // Flag to indicate if the count `cnt` itself is a lucky number.
  
  cin >> n; // Reads the input number.

  // Block Logic: Counts the occurrences of '4's and '7's in the input number `n`.
  // Invariant: After this loop, `cnt` holds the total number of lucky digits found in `n`.
  while (n > 0) {
    // Checks if the last digit is 4 or 7.
    // Inline: `n % 10` extracts the last digit.
    if (n % 10 == 4 || n % 10 == 7)
      cnt++; // Increments the lucky digit counter.
    n /= 10; // Removes the last digit by integer division.
  }

  // Block Logic: Checks if the `cnt` (the count of lucky digits) itself is composed
  //              only of lucky digits (4s and 7s).
  // Invariant: `lucky` remains true only if all digits of `cnt` are either 4 or 7.
  while (cnt > 0) {
    // Checks if the last digit of `cnt` is NOT 4 and NOT 7.
    // Inline: `cnt % 10` extracts the last digit of `cnt`.
    if (cnt % 10 != 4 && cnt % 10 != 7) {
      lucky = false; // If an unlucky digit is found, set flag to false.
      break;         // No need to check further, so exit loop.
    }
    cnt /= 10; // Removes the last digit of `cnt`.
  }

  // Block Logic: Prints the final result based on the `lucky` flag.
  if (lucky)
    cout << "YES"; // Prints "YES" if the original number is "nearly lucky".
  else
    cout << "NO";  // Prints "NO" otherwise.

  return 0; // Indicates successful program termination.
}
