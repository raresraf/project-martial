#include <cstring>
#include <iostream>
using namespace std;

int main() {

  char s[101];
  int val[26];
  for (int i = 0; i < 26; i++)
    val[i] = 0;
  cin >> s;
  int len = strlen(s);
  int c = 0;
  for (int i = 0; i < len; i++) {
    if (val[s[i] - 97] == 0)
      c++;
    val[s[i] - 97]++;
  }
  if (c % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}