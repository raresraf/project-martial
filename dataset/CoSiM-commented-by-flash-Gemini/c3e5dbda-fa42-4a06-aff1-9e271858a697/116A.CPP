/**
 * @file 116A.CPP
 * @brief This program simulates passenger movements (exits and entries)
 *        at a train/bus station over 'n' stops. It calculates and outputs
 *        the minimum required capacity for the vehicle, which corresponds
 *        to the maximum number of passengers present at any single point in time.
 * Algorithm: Simulation, Maximum tracking.
 * Time Complexity: O(N) where N is the number of stops, due to a single linear pass.
 * Space Complexity: O(1) for storing current passenger count and maximum capacity.
 */
#include <algorithm>
#include <fstream>
#include <iostream>

using namespace std;

// Global flag to enable debug mode, which redirects input from "input.txt".
bool DEBUG = false;

// File stream object used when DEBUG is true to read from a file.
ifstream fileStream;

/**
 * @brief Returns an input stream based on the DEBUG flag.
 * @return `fileStream` if DEBUG is true (reads from "input.txt"), otherwise `cin` (standard input).
 * Functional Utility: Provides a flexible input source, allowing easy switching
 *                     between standard input and a debug input file.
 * Pre-condition: If DEBUG is true, "input.txt" must exist in the same directory as the source file.
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
 * @brief Solves the train/bus capacity problem.
 * Functional Utility: Reads passenger exit/entry data, simulates passenger movement,
 *                     and determines the maximum number of passengers on board at any time.
 */
void solve() {
  istream &inputStream = getInputStream(); // Get the appropriate input stream.

  // Your code goes here

  int n; // Number of stops.
  inputStream >> n; // Reads the number of stops.

  int exitNum, enterNum; // Number of passengers exiting and entering at a stop.
  int passengers = 0, capacity = 0; // 'passengers' is current count, 'capacity' is max observed.
  /**
   * Block Logic: Iterates through each stop, updates the current number of passengers,
   *              and tracks the maximum number of passengers seen so far.
   * Invariant: 'passengers' reflects the current count of people on the vehicle,
   *            'capacity' stores the highest 'passengers' value encountered.
   */
  for (int i(0); i < n; i++) {
    inputStream >> exitNum >> enterNum; // Reads passengers exiting and entering.
    passengers = passengers - exitNum + enterNum; // Updates current passenger count.
    if (passengers > capacity) { // If current passengers exceed recorded maximum.
      capacity = passengers;     // Update the maximum capacity.
    }
  }
  cout << capacity << endl; // Prints the calculated minimum required capacity.
}

/**
 * @brief Main function: handles debug argument and calls the solve function.
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * Functional Utility: Entry point of the program; sets debug mode if "debug" argument is provided.
 */
int main(int argc, char *argv[]) {
  if (argc >= 2) { // Checks if a command-line argument is provided.
    string debug(argv[1]); // Converts the first argument to a string.
    if (debug == "debug") { // If the argument is "debug".
      DEBUG = true;         // Enables debug mode.
    }
  }
  solve(); // Calls the core problem-solving logic.
  return 0;
}
