#include <bits/stdc++.h>
using namespace std;

int CountUniqueCharacters(string str) {
  int counter = 0;
  for (int i = 0; i < str.size(); i++) {
    bool appears = false;
    for (int j = 0; j < i; j++) {
      if (str[j] == str[i]) {
        appears = true;
        break;
      }
    }
    if (!appears)
      counter++;
  }

  return counter;
}

int main() {
  string s;
  bool gender;

  cin >> s;

  gender = CountUniqueCharacters(s) % 2;

  if (!gender)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}

/**
wjmzbmr
output:
CHAT WITH HER!

xiaodao
output:
IGNORE HIM!

sevenkplus
output:
CHAT WITH HER!

**/
