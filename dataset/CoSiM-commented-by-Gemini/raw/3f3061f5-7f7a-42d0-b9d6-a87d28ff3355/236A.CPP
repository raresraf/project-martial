#include <iostream>
using namespace std;
int main() {
  string s;
  int flag, c = 0;
  cin >> s;

  for (int i = 'a'; i <= 'z'; i++) {
    flag = 0;
    for (int j = 0; j < s.length(); j++)
      if (s[j] == i)
        flag = 1;
    if (flag == 1)
      c++;
  }
  if (c % 2)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";
}
