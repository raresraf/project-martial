/**
 * @file 282A.CPP
 * @brief Solution for the "Bit++" programming problem.
 *
 * @problem The program calculates the final value of a variable 'x' which starts at 0.
 * A series of 'n' operations are performed, each being one of: "++X", "X++", "--X", "X--".
 *
 * @algorithm The algorithm initializes a variable 'x' to 0. It then iterates 'n' times,
 * reading one operation string in each iteration. It checks if the string represents
 * an increment (contains '++') or a decrement (contains '--'). Based on the
 * operation, it modifies 'x' accordingly. Finally, it prints the final value of 'x'.
 * A simpler way to check the operation is to inspect the middle character of the
 * 3-character string, which is always either '+' or '-'.
 *
 * @time_complexity O(N), where N is the number of operations, as the program
 * iterates through the operations once.
 * @space_complexity O(1), as storage is constant regardless of N (the string is reused).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  int n;
  // Initialize the variable x to 0.
  int x = 0;

  cin >> n;

  string s;

  // Loop n times to process each operation.
  while (n--) {
    cin >> s;
    // Check if the operation is an increment.
    if (s == "++X" || s == "X++")
      x++;
    else // Otherwise, it's a decrement.
      x--;
  }
  
  // Print the final value of x.
  cout << x << endl;

  return 0;
}
