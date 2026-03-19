/**
 * @file 116A.CPP
 * @brief Solution for the "Tram" problem (Codeforces 116A) with a debug-friendly setup.
 * @details This program calculates the minimum required capacity of a tram by simulating
 * passenger flow. It includes a feature to switch input from stdin to a local
 * file ("input.txt") by passing "debug" as a command-line argument.
 *
 * Algorithm: Peak Occupancy Tracking
 * The core logic resides in the `solve()` function. It simulates the passenger
 * count stop-by-stop, maintaining a `passengers` variable for the current count
 * and a `capacity` variable for the maximum count observed. At each stop, `passengers`
 * is updated, and `capacity` is adjusted if a new peak is reached. This is an
 * optimal approach.
 *
 * Time Complexity: O(N), where N is the number of stops.
 * Space Complexity: O(1).
 */
#include <algorithm>
#include <fstream>
#include <iostream>

using namespace std;

// A global flag to control the input source.
bool DEBUG = false;

// A global file stream object for reading from a file in debug mode.
ifstream fileStream;

/**
 * @brief Gets the appropriate input stream based on the DEBUG flag.
 * @return A reference to `std::cin` if DEBUG is false, or a reference to a
 *         `std::ifstream` for "input.txt" if DEBUG is true.
 */
istream &getInputStream() {
  if (DEBUG) {
    string path = __FILE__;
    path = path.substr(0, 1 + path.find_last_of('/'));
    path += "input.txt";
    fileStream.open(path);
    return fileStream;
  }
  return cin;
}

/**
 * @brief Contains the core logic for solving the Tram problem.
 */
void solve() {
  // Obtain the input stream (either cin or a file).
  istream &inputStream = getInputStream();

  // Your code goes here

  // n: the number of stops.
  int n;
  inputStream >> n;

  // exitNum, enterNum: passengers exiting and entering at a stop.
  int exitNum, enterNum;
  // passengers: the current number of people on the tram.
  // capacity: the maximum number of people seen so far.
  int passengers = 0, capacity = 0;
  
  /**
   * Block Logic: Simulate the passenger flow for 'n' stops.
   * Invariant: `capacity` holds the maximum value of `passengers` seen in previous iterations.
   */
  for (int i(0); i < n; i++) {
    inputStream >> exitNum >> enterNum;
    // Update the current passenger count.
    passengers = passengers - exitNum + enterNum;
    // If the current count is a new maximum, update the capacity.
    if (passengers > capacity) {
      capacity = passengers;
    }
  }
  // Output the required minimum capacity.
  cout << capacity << endl;
}

/**
 * @brief The main entry point of the program.
 * @details This function handles command-line argument parsing to enable debug
 * mode and then calls the main solver function.
 */
int main(int argc, char *argv[]) {
  // Check for the "debug" command-line argument.
  if (argc >= 2) {
    string debug(argv[1]);
    if (debug == "debug") {
      DEBUG = true;
    }
  }
  
  // Run the main problem-solving logic.
  solve();
  
  return 0;
}