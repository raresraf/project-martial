#include <bits/stdc++.h>

using namespace std;

set<char> sc;

int main() {

  string s;

  cin >> s;

  for (int i = 0; i < s.length(); ++i)
    sc.insert(s[i]);

  if (sc.size() % 2 == 0)
    printf("CHAT WITH HER!\n");
  else
    printf("IGNORE HIM!\n");
}