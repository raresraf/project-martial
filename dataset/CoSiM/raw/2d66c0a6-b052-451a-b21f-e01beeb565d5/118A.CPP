#include <bits/stdc++.h>
using namespace std;

bool isVowel(char);
string strTask(string &);

int main(void) {
  string str;
  string res;

  cin >> str;

  res = strTask(str);

  cout << res << "\n";

  return 0;
}

bool isVowel(char x) {
  if (x == 'a' || x == 'A' || x == 'e' || x == 'E' || x == 'i' || x == 'I' ||
      x == 'o' || x == 'O' || x == 'u' || x == 'U' || x == 'y' || x == 'Y')
    return true;
  else
    return false;
}

string strTask(string &str) {
  string res;

  for (int i = 0; i < str.length(); i++) {
    if (isVowel(str[i]) == false) {
      res.push_back('.');

      if (str[i] >= 65 && str[i] <= 90)
        res.push_back(str[i] + 32);
      else
        res.push_back(str[i]);
    }
  }

  return res;
}
