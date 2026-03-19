/**
 * @file raw/c6f347ec-95e5-4c48-8e8e-fe0eb0016426/116A.CPP
 * @brief Calculates the minimum required capacity of a tram by simulating passenger flow.
 *
 * This program determines the maximum number of passengers on a tram at any
 * given time, which represents the minimum capacity required. This version
 * separates the core logic into a dedicated function and uses a struct to
 * represent the data for each stop.
 *
 * @time_complexity O(N), where N is the number of stops, due to linear scans
 *                  for input reading and calculation.
 * @space_complexity O(N), as it stores the data for all `n` stops in an array.
 */
#include <iostream>

using namespace std;

/**
 * @struct trum
 * @brief Holds the passenger data for a single tram stop.
 * @note The name is likely a typo for "tram".
 */
typedef struct {
  int entry;
  int exit;
} trum;

/**
 * @brief Calculates the peak number of passengers based on stop data.
 * @param data An array of `trum` structs, each containing entry/exit counts.
 * @param length The number of stops (the length of the `data` array).
 * @return The maximum number of passengers on the tram at any one time.
 *
 * This function iterates through each stop, updating a running total of the
 * current passengers and tracking the maximum value this total reaches.
 */
int max_passengers(trum *data, int length) {
  int last_stop_passengers = 0; // Current passenger count.
  int max_passengers = 0;       // Peak passenger count found so far.

  for (int i = 0; i < length; i++) {
    // Update current passengers: add entries, subtract exits.
    last_stop_passengers =
        (last_stop_passengers - data[i].exit) + data[i].entry;

    // If the current count is a new peak, update the maximum.
    if (max_passengers < last_stop_passengers)
      max_passengers = last_stop_passengers;
  }

  return max_passengers;
}

int main() {
  int n;
  cin >> n;
  // Note: Uses a Variable Length Array (VLA), a non-standard C++ extension.
  trum data[n];

  // Read all stop data into the array of structs.
  for (int i = 0; i < n; i++)
    cin >> data[i].exit >> data[i].entry;

  // Calculate and print the result.
  cout << max_passengers(data, n) << endl;
}
