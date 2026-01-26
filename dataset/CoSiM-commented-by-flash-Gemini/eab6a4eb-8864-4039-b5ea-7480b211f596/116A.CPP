#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, a, b, totalPassenger = 0, maxPassenger = INT_MIN;
  cin >> n;
  while (n--) {
    cin >> a >> b;
    totalPassenger = (totalPassenger - a) + b;
    if (totalPassenger > maxPassenger)
      maxPassenger = totalPassenger;
  }
  cout << maxPassenger;
}
