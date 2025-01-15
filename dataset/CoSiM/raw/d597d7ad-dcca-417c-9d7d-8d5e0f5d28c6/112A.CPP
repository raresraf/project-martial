#include <bits/stdc++.h>

using namespace std;

int main() {
  string a, b;
  cin >> a >> b;
  int check = 0;
  for (int i = 0; i < a.length(); i++) {
    if (tolower(a[i]) > tolower(b[i])) {
      check = 1;
      break;
    } else if (tolower(a[i]) < tolower(b[i])) {
      check = -1;
      break;
    }
  }
  cout << check;
}