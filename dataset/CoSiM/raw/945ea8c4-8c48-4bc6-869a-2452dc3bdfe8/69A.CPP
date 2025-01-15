#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  int x[100];
  int n;
  cin >> n;

  int result = 0, result1 = 0, result2 = 0;

  while (n--) {
    for (int i = 0; i < 3; i++) {
      cin >> x[i];
    }
    result = result + x[0];
    result1 = result1 + x[1];
    result2 = result2 + x[2];
  }

  if (result == 0 && result1 == 0 && result2 == 0) {
    cout << "YES";
  }

  else {
    cout << "NO";
  }

  return 0;
}
