/*
Link: https://codeforces.com/problemset/problem/116/A
@uthor: chavodon
*/

#include <iostream>

using namespace std;

int main() {
  int n, a, b, i, currentPassengers = 0, maxPassengers = 0;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> a >> b;
    currentPassengers -= a;
    currentPassengers += b;
    if (currentPassengers > maxPassengers)
      maxPassengers = currentPassengers;
  }
  cout << maxPassengers;
}
