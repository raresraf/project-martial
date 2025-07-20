#include <bits/stdc++.h>
using namespace std;
int main() {
  string x;
  cin >> x;
  map<char, int> m;
  for (char ch : x)
    m[ch]++;
  int s = m.size();
  if (s % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
  return 0;
}