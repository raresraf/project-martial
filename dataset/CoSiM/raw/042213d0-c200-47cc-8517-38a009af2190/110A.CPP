#include <bits/stdc++.h>
using namespace std;

int main() {
  string str;
  cin >> str;

  int numOfLucky = 0;
  int len = str.size();
  for (int i = 0; i < len; i++) {
    if (str.at(i) == '4' || str.at(i) == '7')
      numOfLucky++;
  }

  string lucky = to_string(numOfLucky);

  bool luck = true;
  len = lucky.size();
  for (int i = 0; i < len; i++) {
    if (lucky.at(i) != '4' && lucky.at(i) != '7')
      luck = false;
  }

  if (luck)
    cout << "YES\n";
  else
    cout << "NO\n";

  return 0;
}
