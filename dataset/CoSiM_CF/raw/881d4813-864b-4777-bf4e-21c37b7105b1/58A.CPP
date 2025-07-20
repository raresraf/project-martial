#include <bits/stdc++.h>
using namespace std;
bool IfCharInString(char s, string str) {
  bool appear = false;

  for (int i = 0; i < str.size(); i++) {
    if (str[i] == s) {
      appear = true;
      break;
    }
  }
  return appear;
}

string FindHelloFromALine(string str) {
  char previous = ' ';
  short int count_l = 0;
  string temp;

  for (int i = 0; i < str.size(); i++) {
    if (str[i] == 'h' || str[i] == 'e' || str[i] == 'l' || str[i] == 'o') {
      if (str[i] == 'l')
        count_l++;

      if ((!IfCharInString(str[i], temp)) || (str[i] == 'l' && count_l == 2)) {
        temp += str[i];
      }
    }
    previous = str[i];
  }

  return temp;
}

int main() {
  string s;

  cin >> s;

  string output = FindHelloFromALine(s);

  if (output.size() == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}

/**
ahhellllloou
output:
YES

hlelo
output:
NO

pnnepelqomhhheollvlo
output:
YES
**/
