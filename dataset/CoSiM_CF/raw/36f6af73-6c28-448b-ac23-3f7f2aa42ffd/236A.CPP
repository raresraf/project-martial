#include <bits/stdc++.h>
using namespace std;
int main() {
  char s[100];
  cin >> s;
  set<char> s1(s, s + strlen(s));
  if (s1.size() % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else
    cout << "IGNORE HIM!";
}
