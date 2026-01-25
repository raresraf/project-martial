#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <map>
#include <math.h>
#include <set>
#include <stdio.h>
#include <string>

using namespace std;

int main() {
  string username;
  cin >> username;
  set<char> letters;
  for (auto i = 0; i < username.length(); i++) {
    letters.insert(username[i]);
  }
  if (letters.size() % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
  return 0;
}