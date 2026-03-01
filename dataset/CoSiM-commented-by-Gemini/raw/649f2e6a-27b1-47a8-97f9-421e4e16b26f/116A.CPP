/**
 * @file 116A.CPP
 * @brief Optimal solution for the "Tram" problem.
 *
 * @problem A tram makes 'n' stops. At each stop, a number of passengers exit
 * and another number enter. The tram starts empty. The task is to find the
 * minimum capacity the tram must have. This is equivalent to finding the
 * maximum number of passengers on the tram at any point in time.
 *
 * @algorithm This code uses an optimal O(N) time and O(1) space approach by
 * processing the data for each stop on the fly.
 * 1. It maintains two key variables: `total` for the current passenger count and
 *    `maximum` for the peak passenger count seen so far.
 * 2. It loops 'n' times, once for each stop.
 * 3. In each iteration, it updates `total` by subtracting the number of exiting
 *    passengers and adding the number of entering passengers.
 * 4. After each update, it compares `total` with `maximum` and updates `maximum`
 *    if the current total is higher.
 * 5. The final value of `maximum` is the answer.
 *
 * @time_complexity O(N), where N is the number of stops, due to the single loop.
 * @space_complexity O(1).
 */
#include <iostream>

using namespace std;

int main() {
  int stops = 0;
  int enter = 0;
  int exit = 0;
  // `maximum` will store the peak passenger count, i.e., the required capacity.
  int maximum = 0;
  // `total` will store the current number of passengers on the tram.
  int total = 0;
  
  cin >> stops;
  
  // Loop through each stop to simulate the passenger flow.
  for (int i = 0; i < stops; i++) {
    cin >> exit;
    cin >> enter;
    
    // Update the current number of passengers on the tram.
    total = total + enter - exit;
    
    // If the current number of passengers is a new peak, update the maximum.
    if (maximum < total) {
      maximum = total;
    }
  }
  
  cout << maximum;
  return 0;
}
