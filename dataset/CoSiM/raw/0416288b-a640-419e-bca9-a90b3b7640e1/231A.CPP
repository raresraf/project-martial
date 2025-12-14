#include <bits/stdc++.h>

using namespace std;

int main() {

  int n;
  cin >> n;

  int counter = 0;
  int smallcounter = 0;

  for (int i = 0; i < n; i++) {

    smallcounter = 0;
    int a, b, c;
    cin >> a >> b >> c;

    if (a == 1) {
      smallcounter++;
    }

    if (b == 1) {
      smallcounter++;
    }

    if (c == 1) {
      smallcounter++;
    }

    if (smallcounter >= 2) {
      counter++;
    }
  }

  cout << counter;
}
