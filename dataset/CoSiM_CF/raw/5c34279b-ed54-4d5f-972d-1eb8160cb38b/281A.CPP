#include <bits/stdc++.h>
using namespace std;
#define ll long long int
int main() {
  string s;
  cin >> s;
  if (s[0] >= 65 && s[0] <= 90)
    cout << char(s[0]);
  else
    cout << char(s[0] - 32);
  for (int i = 1; i < s.length(); i++)
    cout << s[i];
}
