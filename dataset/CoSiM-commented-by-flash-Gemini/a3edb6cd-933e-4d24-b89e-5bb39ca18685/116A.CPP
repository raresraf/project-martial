#include <iostream>

using namespace std;

int main() {
  int n, a, b;
  int passengers = 0;
  int capacity = 0;
  cin >> n;

  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    passengers -= a;
    passengers += b;
    if (passengers > capacity)
      capacity = passengers;
  }

  cout << capacity;

  return 0;
}