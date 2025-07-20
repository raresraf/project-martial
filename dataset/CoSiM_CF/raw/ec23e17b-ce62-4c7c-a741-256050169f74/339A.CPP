#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  string s;
  getline(cin, s);
  int a[s.length() / 2 + 1];
  int j = 0;
  for (int i = 0; i < s.length(); i = i + 2) {
    a[j++] = s[i] - '0';
  }
  sort(a, a + (sizeof(a) / sizeof(a[0])));
  string w = "";
  for (j = 0; j < sizeof(a) / sizeof(a[0]) - 1; j++) {
    w = w + (char)(a[j] + '0') + "+";
  }
  w = w + (char)(a[sizeof(a) / sizeof(a[0]) - 1] + '0');
  cout << w;
}