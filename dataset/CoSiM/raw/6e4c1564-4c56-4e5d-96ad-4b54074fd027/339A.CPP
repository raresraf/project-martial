#include <bits/stdc++.h>
using namespace std;

int main() {

  string s;
  cin >> s;
  int array[100];
  int c = 0;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != '+')
      array[c++] = s[i] - 48;
  }
  sort(array, array + c);
  for (int i = 0; i < c; i++) {
    cout << array[i];
    if (i == c - 1)
      break;
    else
      cout << "+";
  }

  return 0;
}
