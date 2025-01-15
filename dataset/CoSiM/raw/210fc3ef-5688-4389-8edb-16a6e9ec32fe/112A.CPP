// Problem Link https://codeforces.com/problemset/problem/112/A

#include <iostream>

using namespace std;

int main() {
  string a, b;
  cin >> a >> b;
  char c1, c2;
  int result = 0;
  for (int i = 0; i < a.length(); i++) {
    c1 = tolower(a[i]);
    c2 = tolower(b[i]);
    if (c1 < c2) {
      result = -1;
      break;
    } else if (c1 > c2) {
      result = 1;
      break;
    }
  }

  cout << result << "\n";
}