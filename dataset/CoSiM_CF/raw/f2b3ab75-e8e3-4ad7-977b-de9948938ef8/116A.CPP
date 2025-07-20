#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, a, b, capacity = 0, max = 0;
  cin >> n;
  while (n) {
    cin >> a >> b;
    capacity += b - a;

    max = (capacity > max) ? capacity : max;

    n--;
  }

  cout << max;
}
