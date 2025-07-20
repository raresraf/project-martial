#include <bits/stdc++.h>
using namespace std;
int main() {
  int l, c = 0;
  string a;
  char temp;
  cin >> a;
  l = a.length();
  for (int i = 0; i < l - 1; i++) {
    for (int j = i + 1; j < l; j++) {
      if (a[i] > a[j]) {
        temp = a[i];
        a[i] = a[j];
        a[j] = temp;
      }
    }
  }
  for (int i = 0; i < l; i++) {
    if (a[i] == a[i + 1]) {
      continue;
    } else {
      c++;
    }
  }
  if (c % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else {
    cout << "IGNORE HIM!" << endl;
  }
}
