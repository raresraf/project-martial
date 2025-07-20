#include <iostream>

using namespace std;

bool b[30];

int main() {
  int ans = 0;
  string s;
  cin >> s;
  for (int i = 0; i < s.size(); ++i)
    b[s[i] - 'a'] = true;
  for (int i = 0; i < 30; ++i)
    ans += b[i];
  if (ans % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
}