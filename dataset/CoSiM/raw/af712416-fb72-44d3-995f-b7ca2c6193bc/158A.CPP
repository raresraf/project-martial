#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  int n, k;
  int x[100];
  cin >> n >> k;

  int count = 0;
  for (int i = 0; i < n; i++) {
    cin >> x[i];
  }

  for (int i = 0; i < n; i++) {

    if (x[i] >= x[k - 1]) {
      count++;
    } else {
      break;
    }
  }
  int count1 = 0;

  for (int i = 0; i < count; i++) {
    if (x[i] == 0 || x[i] < 0) {
      count1++;
    }
  }
  cout << count - count1;
}
