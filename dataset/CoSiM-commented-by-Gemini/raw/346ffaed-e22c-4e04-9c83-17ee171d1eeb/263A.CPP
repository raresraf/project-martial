/**
 * @file 263A.CPP
 * @brief Converts a series of decimal integers to their binary representation.
 * @note The active code has a bug in its binary-to-string conversion,
 *       appending raw integer values (0 and 1) as character codes instead of
 *       the characters '0' and '1'.
 *
 * This program reads an integer N, followed by N integers. For each of these
 * integers, it computes and prints its binary equivalent.
 *
 * Algorithm (Intended):
 * 1. Handle the special case where X is 0.
 * 2. If X is not 0, repeatedly perform modulo 2 and integer division by 2 to get the binary digits in reverse order.
 * 3. Store these digits in a string.
 * 4. Print the string in reverse to get the correct binary representation.
 *
 * Time Complexity: O(N * log(X_max)), where N is the number of integers to convert,
 * and X_max is the maximum value among them.
 * Space Complexity: O(log(X_max)) to store the binary string for the largest integer.
 */


// The following code block is commented out, but it appears to solve the
// "Beautiful Matrix" problem. The goal is to find the number of moves
// (Manhattan distance) required to move the single '1' in a 5x5 grid to
// the center (3,3).
//
// #include <iostream>
// using namespace std;
//
// int min_moves(int matrix[6][6])
// {
//     int r = 0, c = 0;
//     bool flag = false;
//     for (int i = 1; i <= 5; i++)
//     {
//         for (int j = 1; j <= 5; j++)
//         {
//             if (matrix[i][j] == 1)
//             {
//                 r = i;
//                 c = j;
//                 break;
//                 flag = true;
//             }
//         }
//         if(flag)
//             break;
//     }
//     return abs(r-3)+abs(c-3);
// }
//
// int main()
// {
//     int matrix[6][6];
//     for (int i = 1; i <= 5; i++)
//         for (int j = 1; j <= 5; j++)
//             cin >> matrix[i][j];
//
//     cout << min_moves(matrix);
// }

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/

int main() {
  int N; // Number of integers to convert.
  cin >> N;
  cin.ignore(); // Consume the rest of the line.

  /**
   * Block Logic: Main processing loop.
   * Iterates N times to convert each integer.
   */
  for (int i = 0; i < N; i++) {
    int X; // The integer to be converted.
    cin >> X;
    cin.ignore();
    string s = "";

    if (X == 0) {
      cout << X;
    } else {
      /**
       * Block Logic: Decimal to binary conversion.
       * This loop builds a binary string in reverse.
       */
      while (X) {
        // @note Bug: `X % 2` (an int) is added to the string, which appends
        // the character with that ASCII value (e.g., NULL for 0), not the
        // character '0' or '1'.
        s += X % 2;
        X = X / 2;
      }

      /**
       * Block Logic: Print the binary string in correct order.
       * Because of the bug above, this prints a sequence of non-printable characters.
       */
      for (int i = s.length() - 1; i >= 0; i--) {
        cout << s[i];
      }
    }
    cout << endl;
  }

  // Write an answer using cout. DON'T FORGET THE "<< endl"
  // To debug: cerr << "Debug messages..." << endl;

  // cout << "ones and zeros" << endl;
}