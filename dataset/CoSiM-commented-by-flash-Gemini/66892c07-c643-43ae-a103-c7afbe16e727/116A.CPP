/**
 * @66892c07-c643-43ae-a103-c7afbe16e727/116A.CPP
 * @brief This program solves problem 116A, "Tram".
 * It simulates a tram's journey through a series of stops. For each stop,
 * it tracks the number of passengers exiting and entering, calculates the
 * current number of passengers on board, and continuously updates the
 * maximum number of passengers observed at any point during the journey.
 *
 * Algorithm: Linear simulation. It iteratively updates the current passenger count
 * after each exit and entry, and continuously tracks the maximum observed count.
 * Time Complexity: O(N) where N is the number of tram stops.
 * Space Complexity: O(1) for storing variables.
 */
#include <iostream> // For standard input/output operations (cin, cout).
using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to simulate a tram's passenger count and find the maximum.
 * Reads the number of stops, then for each stop, reads passengers exiting and entering,
 * updates current and maximum passenger counts.
 * @return 0 implicitly, as there is no explicit return statement at the end of main.
 */
int main() {
  int n; // Declares 'n', representing the total number of tram stops.
  cin >> n; // Reads 'n' from standard input.
  
  int entry = 0; // Accumulates the current number of passengers in the tram.
  int exit = 0; // Declared but unused directly for cumulative exit count.
  int inputentry, inputexit; // Stores passengers entering and exiting for the current stop.
  int min = 0; // 'min' variable is used to store the maximum passenger count, which is confusingly named.
  
  /**
   * Block Logic: Simulates the tram's journey through each of the 'n' stops.
   * For each stop, it reads passengers exiting and entering, updates the current
   * passenger count (`entry`), and then updates the maximum observed (`min`)
   * if the current count is higher.
   * Invariant: `min` correctly holds the highest passenger count encountered so far.
   */
  while (n--) { // Loops 'n' times, decrementing 'n' in each iteration.
    // Reads passengers exiting (`inputexit`) and entering (`inputentry`) for the current stop.
    cin >> inputexit >> inputentry; 

    entry -= inputexit; // Inline: Decrements current passenger count by those exiting.
    entry += inputentry; // Inline: Increments current passenger count by those entering.
    
    // Inline: If the current passenger count (`entry`) is greater than the recorded maximum (`min`).
    if (entry > min) 
      min = entry; // Updates 'min' to the new highest count.
  }
  cout << min; // Prints the maximum number of passengers observed.
}