#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  getline(cin, s);
  bool swapped;
  if (s.length() < 3)
    cout << s[0];
  for (int i = 0; i < s.length() - 2; i += 2) {
    swapped = false;
    for (int j = 0; j < s.length() - 2; j += 2)
      if (s[j] > s[j + 2]) {
        swap(s[j], s[j + 2]);
        swapped = true;
      }
    if (swapped == false)
      break;
  }
  cout << s;
  return 0;
}
