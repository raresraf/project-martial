#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, count = 0;
  cin >> n;
  while (n) {
    if (n & 1)
      count++;
    // n >>= 1;
    n = n / 2;
  }
  cout << count << endl;
}
