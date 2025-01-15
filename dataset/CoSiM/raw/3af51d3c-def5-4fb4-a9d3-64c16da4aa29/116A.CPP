#include <bits/stdc++.h>
using namespace std;

int main(void) {
  int n = 0;
  int a = 0;
  int b = 0;
  int passengers = 0;
  int maxPassengers = 0;

  cin >> n;

  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    passengers = passengers - a + b;

    if (passengers > maxPassengers)
      maxPassengers = passengers;
  }

  cout << maxPassengers << "\n";

  return 0;
}
