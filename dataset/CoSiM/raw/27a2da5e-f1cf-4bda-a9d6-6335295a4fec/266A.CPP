/*
Link: https://codeforces.com/problemset/problem/266/A
@uthor: chavodon
*/

#include <iostream>

using namespace std;

int main() {
  int n, i, count = 0;
  string s;
  cin >> n;
  cin >> s;
  for (i = 1; i < n; i++)
    if (s[i - 1] == s[i])
      count++;
  cout << count;
}