/**
 * @file 266B.CPP
 * @brief Solution for the "Queue at the School" problem.
 *
 * @problem In a queue of 'n' children ('B' for boy, 'G' for girl), any boy
 * standing immediately in front of a girl swaps places in one second. All such
 * swaps happen simultaneously. The task is to find the queue's arrangement
 * after 't' seconds.
 *
 * @algorithm The code simulates the process for 't' seconds. In each second (time step),
 * it iterates through the queue. When it finds a 'B' followed by a 'G', it
 * swaps them in-place.
 *
 * A crucial detail is `i++` after a swap. This advances the loop counter an extra
 * step, effectively skipping the next position. This prevents the newly moved 'G'
 * from being considered in a new swap within the same time step, which correctly
 * models the "simultaneous" swap rule for non-overlapping pairs.
 *
 * @time_complexity O(T * N), where T is the number of seconds and N is the
 * number of children.
 * @space_complexity O(N) to store the queue.
 */
#include <iostream>
#include <string.h>

using namespace std;

int main() {
  int n, t;

  cin >> n >> t;

  // Use a char array to store the queue state.
  char a[n];

  cin >> a;

  // Loop for 't' seconds (time steps).
  while (t > 0) {
    // Inner loop scans the queue for 'BG' pairs.
    for (int i = 1; i < n; ++i) {
      if (a[i] == 'G' && a[i - 1] == 'B') {
        // Swap the 'B' and 'G'.
        char aux;
        aux = a[i - 1];
        a[i - 1] = a[i];
        a[i] = aux;
        // Crucial Step: Increment 'i' to skip the next position.
        // This ensures the now-swapped 'G' isn't re-evaluated in this same time step.
        i++;
      }
    }
    t--;
  }

  cout << a << endl;
}
