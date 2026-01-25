#include <algorithm>
#include <iostream>

using namespace std;

#define ll long long;

int n, a[1000], b[1000];

void solve() {
  int max = 0, currentPassengers = 0;
  for (int i = 0; i < n; i++) {
    currentPassengers += b[i] - a[i];
    if (max < currentPassengers)
      max = currentPassengers;
  }
  cout << max;
}

int main() {
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> a[i] >> b[i];
  }
  solve();
}
