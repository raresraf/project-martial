#include <cmath>
#include <iostream>
#include <string>

using namespace std;

void solve() {
  string str;
  cin >> str;
  string x = "hello";
  int j = 0;
  for (int i = 0; i < str.length() && j < x.length(); i++)
    if (str[i] == x[j])
      j++;
  if (j == x.length())
    cout << "YES\n";
  else
    cout << "NO\n";
}

int main() {
  solve();
  return 0;
}