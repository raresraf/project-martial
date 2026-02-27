/**
 * @file 96A.CPP
 * @brief Solution for competitive programming problem: "Football".
 *
 * @details This program analyzes a sequence of '0's and '1's representing football
 *          player positions to determine if a "dangerous" situation occurs. A situation
 *          is considered dangerous if there are 7 or more consecutive players of the
 *          same type (all '0's or all '1's).
 *          Algorithm: Single pass iterative scan with a running counter for consecutive identical characters.
 *          Time Complexity: O(L), where L is the length of the input string, as each
 *                           character is processed exactly once.
 *          Domain: Competitive Programming.
 */
#include <iostream>
using namespace std;

int main() {
  // Define a constant for the minimum number of consecutive identical characters to be considered "dangerous".
  const int dangerous = 7;
  // Declare a string to store the input line of '0's and '1's.
  string line;
  // Read the entire line from standard input.
  getline(cin, line);
  // 'state' stores the current character being counted for consecutive occurrences (-1 indicates no character yet).
  int state = -1;
  // 'runningSum' counts consecutive occurrences of the current 'state' character.
  int runningSum = 0;
  // 'output' stores the final result, initialized to "NO" (not dangerous).
  string output = "NO";
  
  // Block Logic: Iterate through each character of the input string to detect dangerous sequences.
  // Invariant: `runningSum` maintains the count of consecutive identical characters up to `line[k]`.
  for (int k = 0; k < line.size(); k++) {
    // Block Logic: Check if the current character is different from the previous one.
    if (line[k] != state) {
      state = line[k];      // Update the 'state' to the new character.
      runningSum = 0;       // Reset 'runningSum' as a new sequence begins.
    }
    ++runningSum;           // Increment count for the current consecutive character.
    
    // Precondition: If the `runningSum` reaches or exceeds the `dangerous` threshold.
    if (runningSum >= dangerous) {
      output = "YES"; // Set output to "YES" as a dangerous situation is found.
      break;          // Exit the loop early as the condition is met.
    }
  }
  // Output the final result.
  cout << output << endl;
}
