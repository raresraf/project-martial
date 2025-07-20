#include <bits/stdc++.h>
using namespace std;
int main() {
  string c;
  cin >> c;
  int c1 = 0, c2 = 0, c3 = 0;
  for (int i = 0; i < c.length(); i++) {
    if (c[i] == '1')
      c1++;
    if (c[i] == '2')
      c2++;
    if (c[i] == '3')
      c3++;
  }

  string aayush;
  for (int i = 0; i < c1; i++)
    aayush.push_back('1');
  for (int i = 0; i < c2; i++)
    aayush.push_back('2');
  for (int i = 0; i < c3; i++)
    aayush.push_back('3');
  for (int i = 0; i < aayush.length(); i++) {
    cout << aayush[i];
    if (i != aayush.length() - 1)
      cout << '+';
  }

  return 0;
}