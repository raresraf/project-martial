// Implementation.

#include <bits/stdc++.h>
#include <cmath>

using namespace std;

int main(void) {
  int n, count = 0, max = 0, a, b;
  cin >> n;
  while (n--) {
    cin >> a >> b;
    count = count - a + b;
    if (count > max) {
      max = count;
    }
  }
  cout << max << endl;
  return 0;
}