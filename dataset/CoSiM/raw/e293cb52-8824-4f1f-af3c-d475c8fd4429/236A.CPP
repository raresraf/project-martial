/*
Link: https://codeforces.com/problemset/problem/236/A
@uthor: chavodon
*/

#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  int j, k, size, uCount = 0;
  cin >> s;
  size = s.size();
  for (j = 0; j < size; j++)
    for (k = j + 1; k < size; k++)
      if (s[j] == s[k])
        s[k] = '0';
  for (j = 0; j < size; j++)
    if (s[j] != '0')
      uCount++;
  if (uCount % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
}
