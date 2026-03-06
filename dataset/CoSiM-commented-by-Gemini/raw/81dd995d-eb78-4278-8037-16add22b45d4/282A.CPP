/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from Codeforces.
 *
 * This program simulates a series of operations on a variable 'X', which is
 * initialized to 0. It reads a number of operations and then the operations
 * themselves (e.g., "++X", "X++", "--X", "X--"). It calculates and prints the
 * final value of 'X' after all operations are performed.
 *
 * @algorithm Iterative Simulation
 * @complexity
 * Time: O(N), where N is the number of operations. Each string operation is
 *       processed in constant time (string comparison is on short, fixed strings).
 * Space: O(1), as it only uses a few variables to store the state. The string
 *        storage is reused in each loop iteration.
 */
#include <iostream>
#include <string>

/**
 * @brief Main function to handle the simulation.
 */
int main() {
  // n: the number of operations to be performed.
  int n;
  std::cin >> n;
  
  // Consumes the newline character left in the input buffer by `std::cin >> n`,
  // ensuring the subsequent `std::getline` call works correctly.
  std::cin.get();
  
  // x: the variable being incremented or decremented, initialized to 0.
  int x = 0;

  // Pre-condition: 'x' is 0.
  // Block Logic: Loop 'n' times, once for each operation.
  for (int i = 0; i < n; i++) {
    // 'polecenie' is Polish for 'command', it stores the operation string.
    std::string polecenie;
    std::getline(std::cin, polecenie);
    
    // Check the operation string and modify 'x' accordingly.
    if (polecenie == "++X" || polecenie == "X++") {
      x++;
    } else if (polecenie == "--X" || polecenie == "X--") {
      x--;
    }
  }

  // Print the final value of 'x'.
  std::cout << x;

  return 0;
}
