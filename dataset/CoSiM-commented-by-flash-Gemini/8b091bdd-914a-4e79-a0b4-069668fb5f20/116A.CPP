/**
 * @8b091bdd-914a-4e79-a0b4-069668fb5f20/116A.CPP
 * @brief This program simulates passenger flow through a series of train stations
 * and determines the maximum number of passengers present on the train at any single point in time.
 * It processes entry and exit counts for each station to track the current passenger count.
 * Algorithm: Cumulative sum tracking with maximum value detection.
 * Time Complexity: O(N), where N is the number of stations.
 * Space Complexity: O(N) to store the passenger counts at each station.
 */

#include <iostream>
using namespace std;

int main() {
  // @var ar: An array to store the passenger count after each station. Max size 1000.
  int ar[1000];
  // @var enter: Number of passengers entering the train at a station.
  // @var exit: Number of passengers exiting the train at a station.
  // @var r: Current total number of passengers on the train.
  int enter, exit, r = 0;
  // @var station: Total number of stations to process.
  int station;
  
  // Block Logic: Read the total number of stations.
  cin >> station;
  
  // Block Logic: Process passenger flow for each station.
  // Invariant: 'r' holds the current passenger count, and 'ar[i]' stores it after station 'i'.
  // Pre-condition: 'station' defines the number of iterations.
  for (int i = 0; i < station; i++) {
    // Block Logic: Read passengers exiting and entering at the current station.
    cin >> exit >> enter;
    // Inline: Update the current passenger count by subtracting exits and adding entries.
    r = (r + enter) - exit;
    // Inline: Store the passenger count after processing the current station.
    ar[i] = r;
  }
  
  // @var temp: Stores the maximum passenger count found so far.
  int temp = 0;
  
  // Block Logic: Find the maximum passenger count from the recorded values.
  // Invariant: 'temp' stores the highest passenger count encountered up to the current iteration.
  // Pre-condition: 'ar' array contains passenger counts for all stations.
  for (int i = 0; i < station; i++) {
    // Block Logic: If the passenger count at the current station is greater than the current maximum, update 'temp'.
    if (ar[i] > temp) {
      temp = ar[i];
    } 
    // Inline: Continue to the next iteration if the current count is not greater.
    else
      continue;
  }
  
  // Post-condition: Output the maximum passenger count found.
  cout << temp;
  return 0; // Explicitly return 0 for successful execution.
}
