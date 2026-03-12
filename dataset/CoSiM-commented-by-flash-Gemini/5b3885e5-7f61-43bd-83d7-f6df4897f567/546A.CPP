/**
 * @5b3885e5-7f61-43bd-83d7-f6df4897f567/546A.CPP
 * @brief This program calculates the amount of money a soldier needs to borrow.
 * The cost of buying bananas increases linearly: the first costs 'k', the second '2k',
 * and so on, with the 'i'-th banana costing 'i * k' dollars. Given 'w' bananas to buy
 * and 'n' dollars the soldier already has, the program efficiently determines the deficit.
 *
 * Algorithm: Direct mathematical formula for sum of an arithmetic series, followed by
 * subtraction and conditional output.
 * Time Complexity: O(1) as it involves a constant number of arithmetic operations.
 * Space Complexity: O(1) for storing variables.
 */

#include <cstdio>    // For standard C input/output functions like scanf and printf.
using namespace std; // Uses the standard namespace, though not strictly necessary with C I/O.

int main() {
  int k, n, w; // Declares 'k' (cost of first banana), 'n' (initial money), and 'w' (number of bananas).
  int amount;  // Variable to store the calculated amount to borrow.
  
  // Reads 'k', 'n', and 'w' from standard input.
  scanf("%d %d %d", &k, &n, &w); 
  
  // Block Logic: Calculates the amount of money to borrow using a direct formula.
  // The total cost of 'w' bananas is k * (1 + 2 + ... + w).
  // The sum 1 + 2 + ... + w is calculated as w * (w + 1) / 2.
  // Then, 'n' (money owned) is subtracted from the total cost to find the deficit.
  amount = (w * (w + 1) * k / 2) - n; 
  
  // Inline: Uses a ternary operator to print the 'amount' if it's positive,
  // otherwise prints 0 (meaning no borrowing is needed or money is sufficient).
  printf("%d", amount > 0 ? amount : 0); 
  // The program implicitly returns 0 on successful completion of main.
}