#include "string"
#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  string str;
  cin >> str;

  for (int i = 0; i < str.length(); i++) {
    for (int j = i + 1; j < str.length(); j++) {
      if (str[i] == str[j]) {
        str.at(j) = '*';
      }
    }
  }
  int count = 0;
  for (int i = 0; i < str.length(); i++) {
    if (str[i] != '*') {
      count++;
    }
  }
  if (count % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
  return 0;
}
