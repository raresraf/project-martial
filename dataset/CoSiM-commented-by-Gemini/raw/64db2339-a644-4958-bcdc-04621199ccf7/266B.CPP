/**
 * @file 266B.CPP
 * @brief Solution for the "Queue at the School" problem.
 *
 * @problem In a queue of 'n' children ('B' for boy, 'G' for girl), any boy
 * standing immediately in front of a girl swaps places in one second. All such
 * swaps happen simultaneously. The task is to find the queue's arrangement
 * after 'k' seconds.
 *
 * @algorithm The code simulates the process for 'k' discrete time steps. To correctly
 * model the "simultaneous" nature of the swaps, it uses a temporary copy of the
 * string for modifications within each time step.
 * 1. For each second from 1 to 'k':
 * 2. A temporary copy (`z`) of the current queue state (`s`) is created.
 * 3. The code iterates through the original string `s` to identify all 'BG' pairs.
 * 4. When a 'BG' pair is found in `s`, the corresponding characters are swapped
 *    in the temporary string `z`.
 * 5. After checking all positions, the modified copy `z` replaces `s`, becoming
 *    the new state for the next time step.
 *
 * @time_complexity O(K * N), where K is the number of seconds and N is the
 * length of the queue.
 * @space_complexity O(N) to store the queue string and its temporary copy.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k;
  cin >> n >> k;
  string s;
  cin >> s;
  
  // Loop for 'k' seconds (time steps).
  while (k--) {
    // Create a temporary copy to perform swaps on.
    string z = s;
    // Iterate through the original string `s` to find pairs.
    for (int i = 1; i < s.length(); i++)
      // If a 'BG' pair is found in the original state `s`...
      if (s[i - 1] == 'B' && s[i] == 'G')
        // ...perform the swap in the copy `z`.
        swap(z[i - 1], z[i]);
    // Update the main string `s` with the result of this time step's swaps.
    s = z;
  }
  
  cout << s << endl;
  return 0;
}
