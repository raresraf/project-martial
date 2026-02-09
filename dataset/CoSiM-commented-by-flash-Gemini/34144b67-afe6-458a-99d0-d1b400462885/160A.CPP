#include <algorithm> // Required for std::min, std::max, etc. (though not used in this specific snippet)
#include <iostream>  // Required for input/output operations (cin, cout)

using namespace std; // Using the standard namespace to avoid prefixing std::

int main() { // Main function where program execution begins
  int manus, capacity, T, n = 0, v; // Declare integer variables:
                                 // manus: represents a certain quantity (e.g., current items)
                                 // capacity: represents the maximum capacity
                                 // T: number of test cases or iterations
                                 // n: counter, initialized to 0
                                 // v: calculated difference (available space)

  cin >> T; // Read the number of test cases/iterations

  while (T--) { // Loop through each test case, decrementing T after each iteration
    cin >> manus >> capacity; // Read 'manus' and 'capacity' for the current test case

    v = capacity - manus; // Calculate the difference (available space)

    if (v >= 2) { // Check if the available space is 2 or more
      n++;        // If true, increment the counter 'n'
    }
  }

  cout << n << endl; // Print the final count 'n'
  return 0;          // Indicate successful program termination
}