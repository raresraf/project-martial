#include <algorithm>
#include <iostream>
#include <string>
using namespace std;
int main() {
  string s;
  cin >> s;
  sort(s.begin(), s.end());
  int n = 0;
  for (int i = 1; i < s.length(); i++)
    if (s[i] != s[i - 1])
      n++;
  if ((n - 1) % 2 != 0)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";
  return 0;
}
