// CodeForces
// Problem 116A [Tram]

#include <bits/stdc++.h>
using namespace std;

int main() {
  long long total_stops, current_capacity = 0, max_capacity = 0, entries = 0,
                         exits = 0;
  cin >> total_stops;

  while (total_stops--) {
    cin >> exits >> entries;
    current_capacity -= exits;
    current_capacity += entries;
    if (current_capacity > max_capacity) {
      max_capacity = current_capacity;
    }
  }
  cout << max_capacity;
  return 0;
}