#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;
int main() {
  string s;
  cin >> s;
  sort(s.begin(), s.end());
  s.erase(unique(s.begin(), s.end()), s.end());
  if (s.length() % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
}
