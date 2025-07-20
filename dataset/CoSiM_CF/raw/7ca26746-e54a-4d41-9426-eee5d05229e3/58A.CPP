#include <bits/stdc++.h>
using namespace std;
int main() {
  string a, b = "hello";
  int inc = 0;
  int j = 0;
  cin >> a;
  for (int i = 0; i < a.size(); i++) {
    if (a[i] == b[j]) {
      j++;
      inc += 1;
      if (inc == 5) {
        break;
      }
    }
  }
  if (inc == 5) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
