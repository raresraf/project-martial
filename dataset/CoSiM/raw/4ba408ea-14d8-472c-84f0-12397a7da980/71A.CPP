#include <bits/stdc++.h>
using namespace std;

int main() {
  int t;
  cin >> t;
  string temp;
  for (int i = 0; i < t; ++i) {
    cin >> temp;
    int l = temp.length();
    if (l <= 10) {
      cout << temp << endl;
    } else {
      cout << temp[0] << l - 2 << temp[l - 1] << endl;
    }
  }
}